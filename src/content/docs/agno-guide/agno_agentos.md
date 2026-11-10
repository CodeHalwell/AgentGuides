---
title: "Agno: AgentOS"
description: "Serving Agno agents, teams and workflows with AgentOS: the FastAPI app, the REST and SSE API, sessions and user ownership, background runs, MCP serving, chat and protocol interfaces (AG-UI, A2A, Slack, Telegram, WhatsApp), authentication, the control plane, schedules and remote clients."
framework: "agno"
language: "python"
sidebar:
  label: "AgentOS"
  order: 40
---

Checked against **agno %%version=agno%%** (checked %%versions-checked%%). The app, routes, request formats and responses below were exercised with FastAPI's `TestClient` and against a live `uvicorn` server, using scripted stand-in models. Interfaces that need third-party accounts (Slack, Telegram, WhatsApp) were mounted but not called; the hosted control plane was not used.

## What AgentOS is

`AgentOS` turns your agents, teams and workflows into a FastAPI application: about 80 to 95 REST paths (78 for one agent, one team and one workflow with no database; 95 once a `db` is configured, which adds the session, memory, knowledge, eval, trace, metric and schedule routes) for runs, sessions, memories, knowledge, evals, traces, metrics and schedules, with Server-Sent Events for streaming. It runs in your own infrastructure and stores everything in your database. Optional extras on the same app: an MCP server, chat interfaces, JWT authorisation with per-user isolation, a background job queue and a cron scheduler.

Install it with `pip install "agno[os]"` (FastAPI, uvicorn, SQLAlchemy, PyJWT, OpenTelemetry and the scheduler's dependencies).

## A minimal server

```python
# app.py
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.os import AgentOS
from agno.team import Team

db = SqliteDb(db_file="agno.db")

assistant = Agent(
    id="assistant",
    name="Assistant",
    model=OpenAIResponses(id="%%model=openai%%"),
    add_history_to_context=True,
)
desk = Team(id="desk", name="Desk", model=OpenAIResponses(id="%%model=openai%%"), members=[assistant])

agent_os = AgentOS(
    id="my-os",
    description="Support assistants",
    db=db,                     # used by every agent, team and workflow without its own db
    agents=[assistant],
    teams=[desk],
)
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="app:app", reload=True)   # http://localhost:7777
```

`serve()` runs uvicorn (`host="localhost"`, `port=7777` by default; `AGENT_OS_HOST` and `AGENT_OS_PORT` override them, and `workers=` is passed through). Because `app` is an ordinary FastAPI app you can also run it with `uvicorn app:app` yourself. Interactive API docs are at `/docs`.

Give every agent, team and workflow a stable `id`: it is part of the URL. AgentOS needs at least one agent, team, workflow, knowledge base or database.

## Calling the API

Run endpoints take form data (multipart, so files can be attached). `stream` defaults to **true**:

```bash
# Server-Sent Events
curl -N -X POST http://localhost:7777/agents/assistant/runs \
  -F message="What is the capital of Portugal?" \
  -F session_id=chat-1 -F user_id=ada

# One JSON response
curl -X POST http://localhost:7777/agents/assistant/runs \
  -F message="And of Spain?" -F stream=false -F session_id=chat-1 -F user_id=ada
```

A streamed response is a sequence of `event:` / `data:` pairs with the same event names as in Python:

```text
event: RunStarted
data: {"event":"RunStarted","agent_id":"assistant","run_id":"...","session_id":"chat-1","model":"...", ...}

event: RunContent
data: {"event":"RunContent","content":"Lisbon", ...}
...
event: RunCompleted
```

With `stream=false` the body is the serialised `RunOutput` (`content`, `status`, `run_id`, `session_id`, `user_id`, `tools`, `metrics`, …). Teams and workflows work the same way at `/teams/{team_id}/runs` and `/workflows/{workflow_id}/runs`.

Other form fields on the run endpoints: `files` (uploads), `version` (a stored component version) and `background`. Any further field is passed to `run()` as a keyword argument; `session_state`, `dependencies`, `metadata`, `knowledge_filters` and `output_schema` are sent as JSON strings (`-F 'metadata={"source": "web"}'`).

Sessions belong to the `user_id` that created them. A run that names an existing session but a different (or missing) `user_id` is refused with `404 Session not found`, even without authentication configured.

### The main endpoints

| Area | Endpoints |
|------|-----------|
| Discovery | `GET /health`, `GET /info` (unauthenticated metadata), `GET /config` (agents, teams, workflows, databases, available models, interfaces) |
| Agents | `GET /agents`, `GET /agents/{id}`, `POST /agents/{id}/runs`, `GET /agents/{id}/runs/{run_id}`, `POST .../runs/{run_id}/continue` (resume a paused run), `POST .../runs/{run_id}/cancel`, `POST .../runs/{run_id}/resume` (reconnect to a streaming run), `POST /agents/{id}/sessions/{session_id}/fork` and `/compact` |
| Teams, workflows | the same shapes under `/teams` and `/workflows` |
| Sessions | `GET/POST/DELETE /sessions` (filter with `type=agent|team|workflow`, `user_id`, …), `GET/PATCH/DELETE /sessions/{session_id}`, `GET /sessions/{session_id}/runs`, `POST /sessions/{session_id}/rename` |
| Memory | `GET/POST/DELETE /memories`, `GET/PATCH/DELETE /memories/{memory_id}`, `GET /memory_topics`, `POST /optimize-memories` |
| Knowledge | `GET/POST/DELETE /knowledge/content`, `POST /knowledge/search`, `GET /knowledge/config` |
| Evals, traces, metrics | `GET/POST/DELETE /eval-runs`, `GET /traces`, `POST /traces/search`, `GET /metrics`, `POST /metrics/refresh` |
| Operations | `/schedules`, `/approvals` (see [Approvals](/agno-guide/agno_comprehensive_guide/#approvals)), `/queue`, `/service-accounts`, `/components`, `/registry`, `POST /databases/{db_id}/migrate` |

The OpenAPI schema at `/openapi.json` is the authoritative list for your configuration; some routers only appear when the matching feature is enabled.

### Human-in-the-loop over HTTP

A paused run comes back with `status: "PAUSED"`, its `requirements`, and `tools` entries with `requires_confirmation: true` (or `requires_user_input`, `external_execution_required`). Set `confirmed` (or fill the input fields, or `result`) on those entries and post the list back, as a JSON string in the `tools` field, to `/agents/{id}/runs/{run_id}/continue` with the same `session_id` and `user_id`:

The snippet assumes an AgentOS serving an agent with id `ops` that has a tool with `requires_confirmation=True`, such as `delete_file` in the [comprehensive guide](/agno-guide/agno_comprehensive_guide/#confirmation). It was run against a live server with a scripted model.

```python
import json

import httpx

base = "http://localhost:7777"
who = {"session_id": "ops-1", "user_id": "ada", "stream": "false"}

run = httpx.post(f"{base}/agents/ops/runs", data={"message": "Delete /tmp/x", **who}).json()
if run["status"] == "PAUSED":
    tools = run["tools"]
    for t in tools:
        if t.get("requires_confirmation"):
            t["confirmed"] = True
    run = httpx.post(f"{base}/agents/ops/runs/{run['run_id']}/continue",
                     data={"tools": json.dumps(tools), **who}).json()
print(run["status"], run["content"])
```

In Python, `AgentOSClient.continue_agent_run` (below) wraps the second call.

## Background runs and the queue

`background=true` returns immediately with `202` and `status: "PENDING"`; the run executes in the server and you poll `GET /agents/{id}/runs/{run_id}?session_id=...` (or reconnect with `/resume`):

```bash
curl -X POST http://localhost:7777/agents/assistant/runs \
  -F message="Write the weekly report" -F stream=false -F background=true
```

Background runs are capped at 32 at a time per process (`AGNO_BACKGROUND_MAX_CONCURRENCY`), and further runs wait as `PENDING`. `AgentOS(queue=QueueConfig(...))` changes that: `max_concurrency`, `durable=True` (accepted requests are stored as rows that survive restarts and can be executed by any replica, with `Idempotency-Key` support and the `/queue` endpoints), and `redis=` (cross-replica cancellation and event streaming). Durable and Redis modes were not exercised here.

## Serving over MCP

`mcp=True` serves the OS as an MCP server at `/mcp` (Streamable HTTP). The default surface is eight tools: `get_agentos_config`, `run_agent`, `run_team`, `run_workflow`, `continue_run`, `cancel_run`, `get_sessions` and `get_session_runs`. To publish specific components and functions instead, pass an `MCPConfig`:

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.os import AgentOS, MCPConfig

assistant = Agent(id="assistant", name="Assistant", model=OpenAIResponses(id="%%model=openai%%"))


def word_count(text: str) -> int:
    """Count the words in a text."""
    return len(text.split())


agent_os = AgentOS(
    db=SqliteDb(db_file="agno.db"),
    agents=[assistant],
    mcp=MCPConfig(
        tools=[assistant.as_tool(name="ask_assistant", description="Ask the support assistant"), word_count],
        allowed_hosts=["agents.example.com"],   # DNS-rebinding protection; localhost is always allowed
    ),
)
app = agent_os.get_app()
```

`tools/list` then returns exactly `ask_assistant` (arguments `message`, optional `user_id`, `session_id`) and `word_count`. Calling `ask_assistant` runs the agent and returns its answer as text, with `run_id`, `session_id` and `status` as structured content. `as_tool()` markers only work here: putting one in `Agent(tools=[...])` raises a `ValueError` that points you to a `Team` for agent-to-agent delegation.

Other `MCPConfig` fields: `default_tools=True` (add the eight built-ins), `lifecycle_tools=True`, `include_tags` / `exclude_tags` (`"core"`, `"session"`, `"lifecycle"`), `authorize=` (a function of the verified `user_id`, which is `None` when no JWT is configured, so decide what anonymous means), `path`, `instructions`, `middleware`. In testing, a request whose `Host` header was not a localhost name was rejected with `400 invalid_host` even with plain `mcp=True`; list your public host in `allowed_hosts`. `mcp_auth=` adds OAuth for connector clients (not exercised).

## Interfaces

Interfaces mount extra routers on the same app for chat platforms and agent protocols:

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.os import AgentOS
from agno.os.interfaces.a2a import A2A
from agno.os.interfaces.agui import AGUI

assistant = Agent(id="assistant", name="Assistant", model=OpenAIResponses(id="%%model=openai%%"))

agent_os = AgentOS(
    db=SqliteDb(db_file="agno.db"),
    agents=[assistant],
    interfaces=[AGUI(agent=assistant), A2A(agents=[assistant])],
)
app = agent_os.get_app()
```

| Interface | Import | Routes | Needs |
|-----------|--------|--------|-------|
| AG-UI | `agno.os.interfaces.agui.AGUI(agent=... or team=...)` | `POST /agui` (AG-UI event stream; verified with a `RunAgentInput` body), `GET /status` | `agno[agui]` |
| A2A | `agno.os.interfaces.a2a.A2A(agents=[...], teams=[...], workflows=[...])`, or `AgentOS(a2a_interface=True)` | `/a2a/agents/{id}/.well-known/agent-card.json`, `/a2a/agents/{id}/v1/message:send`, `.../message:stream`, `.../tasks:get`, `.../tasks:cancel` (and the same for teams and workflows) | `agno[a2a]` |
| Slack | `agno.os.interfaces.slack.Slack(agent=...)` | `POST /slack/events`, `POST /slack/interactions` | `agno[slack]`, `SLACK_TOKEN`, `SLACK_SIGNING_SECRET` |
| Telegram | `agno.os.interfaces.telegram.Telegram(agent=...)` | `POST /telegram/webhook`, `GET /telegram/status` | `agno[telegram]` (importing the module without it raises `ImportError`), `TELEGRAM_TOKEN` |
| WhatsApp | `agno.os.interfaces.whatsapp.Whatsapp(agent=...)` | `/whatsapp/webhook` (GET verify, POST messages), `GET /whatsapp/status` | `WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_VERIFY_TOKEN` |

Each takes one agent, team or workflow (A2A takes lists) and a `prefix` to change the mount point. This release has no Discord interface module, although the project README mentions Discord.

## Authentication and isolation

With no configuration, every endpoint is open. Three levels:

**A shared key.** Set `OS_SECURITY_KEY` in the environment and every endpoint except `/health` requires `Authorization: Bearer <key>`:

```bash
export OS_SECURITY_KEY="a-long-random-string"
curl http://localhost:7777/agents -H "Authorization: Bearer $OS_SECURITY_KEY"
```

**JWT verification and per-user isolation.** Pass an `Authorization` object with your identity provider's verification keys (or a JWKS file). The token's `sub` becomes the run's `user_id`; with `user_isolation=True` each user sees only their own sessions and memories:

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.os import AgentOS, Authorization, create_dev_token

SECRET = "dev-secret-at-least-256-bits-long-xxxxxxxxxxxxxxxx"   # local testing only

agent_os = AgentOS(
    id="my-os",
    db=SqliteDb(db_file="agno.db"),
    agents=[Agent(id="assistant", model=OpenAIResponses(id="%%model=openai%%"))],
    authorization=Authorization(verification_keys=[SECRET], algorithm="HS256"),
    user_isolation=True,
)
app = agent_os.get_app()

# Mint a token for a pretend user (development only; production tokens come from your IdP)
alice = create_dev_token("alice", secret=SECRET, scopes=["agents:*:read", "agents:*:run"])
# curl -H "Authorization: Bearer $alice" ...
```

Each route checks scopes, so a token minted with only `agents:*:read` and `agents:*:run` can run agents but gets `403 Insufficient permissions. Required scope(s): sessions:read` from `GET /sessions`. Add the scopes for the routes the client needs.

Verified behaviour: requests without a token get `401`; Alice's run is stored with `user_id="alice"`; another user asking for Alice's session gets `404`. For production use an asymmetric algorithm (`RS256`, the middleware default) with your provider's public keys (`verification_keys=`, `jwks_file=`, or the `JWT_VERIFICATION_KEY` / `JWT_JWKS_FILE` environment variables), and set `audience=` / `verify_audience=True`.

**Managed roles.** `Authorization` can also hold roles mapped to scopes (`define_role("runner", ["agents:*:read", "agents:*:run"])`, `seed(admin=...)`, `assign(user, role)`), an audit log (`audit=True`) and an `/authz` admin API. These need a SQL database and were not exercised for this guide. `AgentOS(user_directory=True)` keeps a roster of users (auto-registered from verified tokens).

Service accounts (`agno_pat_...` tokens) can be minted with the `agno tokens` CLI command (from the `agnoctl` package installed alongside `agno`) or the `/service-accounts` API, for machine clients. Not exercised.

## The control plane

When `serve()` starts it prints `https://os.agno.com/` alongside the local address. That is Agno's hosted web UI: you add your AgentOS endpoint to it and it calls your API from the browser, so your data stays in your database. It shows chats, sessions, memories, knowledge, evals, traces, metrics and schedules for your agents. It is a separate product with its own account and plans, and it was not used for this guide. Everything it does goes through the endpoints above, so you can build your own UI on the same API.

The `agno` command (from `agnoctl`) is the terminal side: `agno create` (new project from a template), `agno up` / `down` / `restart` (docker compose), `agno connect` (register the OS as an MCP server with local coding agents), `agno status`, `agno tokens`.

## Schedules

`AgentOS(scheduler=True)` starts a poller that fires stored cron schedules against the OS's own endpoints. Schedules are rows in the database, created through `/schedules` or `ScheduleManager`:

```python
from agno.db.sqlite import SqliteDb
from agno.scheduler import ScheduleManager

schedules = ScheduleManager(SqliteDb(db_file="agno.db"))
digest = schedules.create(
    name="weekday-digest",
    cron="0 8 * * 1-5",
    endpoint="/agents/assistant/runs",
    payload={"message": "Write the morning digest"},
    timezone="Europe/London",
)
print(digest.next_run_at)   # Unix timestamp of the next 08:00 London time on a weekday
```

Other arguments: `method`, `timeout_seconds`, `max_retries`, `retry_delay_seconds`, `if_exists` (`"raise"`, `"skip"`, `"update"`), `user_id`. The poller interval is `scheduler_poll_interval` (15 seconds). Creating schedules was verified; the poller firing them was not.

## Serving external coding agents

AgentOS can also serve agents that are not built from `Agent`. `ClaudeAgent` (`agno.agents.claude`) wraps the Claude Agent SDK, which runs Claude Code as a subprocess, and `CodexAgent` (`agno.agents.codex`) does the same for Codex. The harness runs its own tool loop; Agno adds the run interface, session storage and the AgentOS endpoints. Install the SDK alongside Agno (`pip install claude-agent-sdk`).

```python
from agno.agents.claude import ClaudeAgent
from agno.os import AgentOS

coder = ClaudeAgent(
    name="Claude Coder",
    allowed_tools=["Read", "Edit", "Bash"],  # pre-approved; does not restrict the tool set
    permission_mode="acceptEdits",
    max_turns=10,
    max_budget_usd=1.0,
)

app = AgentOS(agents=[coder]).get_app()  # appears under /agents with id "claude-coder"
```

Standalone, `coder.print_response("Summarise main.py", stream=True)` works as it does for `Agent`. Points to know:

- `tools` selects built-in tools, `allowed_tools` only pre-approves calls, and `disallowed_tools` blocks them. `max_turns` and `max_budget_usd` stop a run that goes on too long.
- `output_schema` takes a Pydantic model; the run content is then the parsed object.
- `cwd` is the working directory the harness sees. Files attached to a run are staged under `.agno/uploads/<run_id>/` and deleted afterwards unless `keep_uploads=True`.
- `require_native_resume=True` fails a run if the earlier native conversation cannot be resumed, instead of replaying the history into a new one.
- `CodexAgent` has similar arguments, plus `sandbox_mode`, `approval_mode` and `reasoning_effort`.

### Running the harness in a sandbox

Pass `sandbox=` to run the whole harness in a container each turn, with the files kept between turns as checkpoints. `DockerSandbox(image)` (`agno.sandbox.docker`) starts containers with no host bind mounts and the default `network="none"`; resource limits are `cpus`, `memory_mb`, `pids_limit` and `workspace_mb`. The harness needs to reach its model API, so use `network="isolated"` (or another network); with `"none"` Agno logs a warning. The agent or its AgentOS also needs a `PostgresDb`, which this page did not exercise.

```python
from agno.agents.claude import ClaudeAgent
from agno.sandbox.docker import DockerSandbox

coder = ClaudeAgent(
    name="Sandboxed Coder",
    sandbox=DockerSandbox("python:3.12-slim", network="isolated", cpus=1, memory_mb=1024),
)
```

`Agent(sandbox=...)` is not supported yet: it raises an error pointing at `ClaudeAgent` and `CodexAgent`. Construction and registration with AgentOS were checked against agno %%version=agno%%; running a harness turn needs Claude Code or Codex credentials and, for a sandbox, Docker and Postgres, so those were not run.

## Adding AgentOS to an existing FastAPI app

Pass your app as `base_app`; AgentOS adds its routes and middleware to it and `get_app()` returns the same object:

```python
from fastapi import FastAPI

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.os import AgentOS

app = FastAPI(title="My product")


@app.get("/hello")
def hello():
    return {"hello": "world"}


agent_os = AgentOS(
    agents=[Agent(id="assistant", model=OpenAIResponses(id="%%model=openai%%"))],
    base_app=app,
    on_route_conflict="preserve_base_app",   # or "preserve_agentos" (default), "error"
)
app = agent_os.get_app()
```

`lifespan=` adds your own startup/shutdown hook; `cors_allowed_origins=` (or `cors=CORSConfig(...)`) sets CORS.

## Calling a remote AgentOS from Python

`RemoteAgent` behaves like an agent whose runs execute on a server; `AgentOSClient` covers the rest of the API. Both are async:

```python
import asyncio

from agno.agent import RemoteAgent
from agno.client import AgentOSClient


async def main():
    remote = RemoteAgent(base_url="http://localhost:7777", agent_id="assistant")
    run = await remote.arun("Hello", user_id="ada", session_id="r1")
    print(type(run).__name__, run.content)       # RunOutput ...

    client = AgentOSClient(base_url="http://localhost:7777")
    print([a.id for a in await client.list_agents()])
    run = await client.run_agent(agent_id="assistant", message="Again", user_id="ada", session_id="r1")


asyncio.run(main())
```

`AgentOS(agents=[...])` also accepts `RemoteAgent` (and `teams=` / `workflows=` accept `RemoteTeam` / `RemoteWorkflow`), so one OS can front another; that was not exercised. `RemoteAgent(protocol="a2a")` talks to any A2A server instead. The client also has `run_agent_stream`, `continue_agent_run`, `cancel_agent_run`, team and workflow equivalents, and session and memory methods.

## AgentOS reference

| Group | Parameters |
|-------|-----------|
| Identity | `id`, `name`, `description`, `version` |
| Components | `agents`, `teams`, `workflows`, `knowledge`, `registry` |
| Storage | `db`, `auto_provision_dbs`, `media_storage`, `checkpoint` |
| Surfaces | `interfaces`, `a2a_interface`, `mcp`, `mcp_auth`, `base_app`, `on_route_conflict`, `lifespan`, `cors_allowed_origins`, `cors` |
| Security | `authorization`, `user_isolation`, `user_directory`, `settings` (`AgnoAPISettings`, reads `OS_SECURITY_KEY`) |
| Execution | `queue`, `event_stream`, `run_hooks_in_background`, `scheduler`, `scheduler_poll_interval`, `scheduler_base_url` |
| Observability | `tracing`, `telemetry` |
| Other | `config` (YAML file or `AgentOSConfig`), `manager`, `company`, `public` |

Methods: `get_app()`, `serve(app, host, port, reload, workers, ...)`, `get_routes()`, `run_diagnostics()`.

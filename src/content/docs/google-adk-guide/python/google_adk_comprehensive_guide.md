---
title: "Google ADK Comprehensive Guide (Python)"
description: "A single-page tour of Google ADK for Python: setup, the adk CLI, models, agents, tools, structured output, multi-agent systems, workflows, state, memory, callbacks, MCP and A2A, Gemini features, live streaming and evaluation."
framework: "google-adk"
language: "python"
sidebar:
  label: "Comprehensive guide"
  order: 10
---

Checked against `google-adk` %%version=google-adk%% on %%versions-checked%%. The Python examples were run against a stub model that stands in for Gemini, so they show working calls and event handling rather than real model output.

This page is the overview: one runnable example per concept, with links to the topic pages for the full detail. Read it top to bottom on your first day with ADK, then use the topic pages as reference.

## Contents

1. [What ADK is](#what-adk-is)
2. [Installation and setup](#installation-and-setup)
3. [Project layout and the adk CLI](#project-layout-and-the-adk-cli)
4. [Models](#models)
5. [Your first agent](#your-first-agent)
6. [Tools](#tools)
7. [Structured output](#structured-output)
8. [Multi-agent systems](#multi-agent-systems)
9. [Workflows](#workflows)
10. [State, sessions and memory](#state-sessions-and-memory)
11. [Instructions and context](#instructions-and-context)
12. [Callbacks and plugins](#callbacks-and-plugins)
13. [MCP and A2A](#mcp-and-a2a)
14. [Gemini features](#gemini-features)
15. [Streaming and live sessions](#streaming-and-live-sessions)
16. [Evaluation and optimisation](#evaluation-and-optimisation)
17. [Deployment](#deployment)
18. [Import paths worth knowing](#import-paths-worth-knowing)

## What ADK is

The Agent Development Kit (ADK) is Google's open-source, code-first framework for building agents. It is tuned for Gemini and Google Cloud but not tied to them: other models plug in through `LiteLlm`, `AnthropicLlm`, the OpenAI classes or your own `BaseLlm`.

The pieces you will meet:

| Piece | What it is |
|---|---|
| `LlmAgent` (alias `Agent`) | An agent driven by a model, with instructions, tools and optional sub-agents |
| `Workflow` | A graph of agents and functions, for deterministic orchestration (ADK 2.x) |
| Tools | Python functions, other agents, Gemini built-ins, MCP servers, OpenAPI specs |
| `Runner` | Executes an agent for one user turn and yields `Event`s |
| `App` | Wraps the root agent with plugins, context compaction, caching and resumability |
| Session, memory and artifact services | Where conversation state, long-term memory and files are stored |
| `adk` CLI | Scaffolding, a terminal runner, the Dev UI, an API server, evaluation and deployment |

ADK 2.0 (May 2026) introduced `Workflow` graph orchestration and deprecated the older `SequentialAgent`, `ParallelAgent` and `LoopAgent`, which still work but emit a `DeprecationWarning`.

## Installation and setup

ADK needs Python 3.11 or later (3.10 support was removed in 2.12.0).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install google-adk
```

Optional integrations are extras. Install the ones you use:

| Extra | Brings in |
|---|---|
| `mcp` | `McpToolset`, `to_mcp_server` |
| `a2a` | `to_a2a`, `RemoteA2aAgent` |
| `eval` | `AgentEvaluator` and the metric suite |
| `db` | SQLAlchemy drivers for `DatabaseSessionService` |
| `extensions` | `LiteLlm`, `AnthropicLlm`, container code executors, the LangGraph bridge |
| `openai` | `OpenAILlm`, `OpenAIResponsesLlm`, `AzureOpenAIResponsesLlm` |
| `gcp` | Cloud Observability exporters, Model Armor |
| `bigquery-analytics` | `BigQueryAgentAnalyticsPlugin` |

For example `pip install "google-adk[mcp,a2a,eval]"`. Pin the version in deployed code: `google-adk==%%version=google-adk%%`.

### Credentials

ADK reaches Gemini either through the Gemini API with an API key, or through Vertex AI with Google Cloud credentials. It reads the choice from environment variables, normally kept in a `.env` file in the agent folder:

```bash
# Gemini API (Google AI Studio key)
GOOGLE_GENAI_USE_ENTERPRISE=0
GOOGLE_API_KEY=your-key

# or Vertex AI (Application Default Credentials: gcloud auth application-default login)
GOOGLE_GENAI_USE_ENTERPRISE=1
GOOGLE_CLOUD_PROJECT=your-project
GOOGLE_CLOUD_LOCATION=europe-west2
```

`GOOGLE_GENAI_USE_ENTERPRISE` is the name `adk create` writes in 2.12; the older `GOOGLE_GENAI_USE_VERTEXAI` is still honoured. The `adk` commands load the agent folder's `.env` for you; in your own scripts, export the variables or use `python-dotenv`. Keep `.env` out of version control (`adk create` adds it to `.gitignore`). For service accounts, least-privilege roles and Secret Manager, see [IAM, auth and credentials](/google-adk-guide/python/google_adk_iam_examples/).

## Project layout and the adk CLI

The `adk` commands work on an **agent folder**: a Python package whose `agent.py` defines a module-level `root_agent`.

```text
my_agent/
├── __init__.py      # from . import agent
├── agent.py         # root_agent = Agent(...)
├── .env             # credentials and settings for local runs
└── .gitignore       # .env and .adk/
```

`adk create` writes exactly this (`--model`, `--api_key`, `--project` and `--region` skip the prompts):

```bash
adk create my_agent --model %%model=gemini%%
```

The generated `agent.py`:

```python
from google.adk.agents.llm_agent import Agent

root_agent = Agent(
    model="%%model=gemini%%",
    name="root_agent",
    description="A helpful assistant for user questions.",
    instruction="Answer user questions to the best of your knowledge",
)
```

The commands you will use most:

| Command | What it does |
|---|---|
| `adk run my_agent` | Chat with the agent in the terminal; `adk run my_agent "question"` runs one turn and exits |
| `adk web .` | The Dev UI on `http://127.0.0.1:8000`: chat, the event and trace view, state, artifacts, eval sets. Pass the folder that *contains* your agent folders |
| `adk api_server .` | The same FastAPI server without the UI, for local integration tests |
| `adk eval my_agent path/to/set.evalset.json` | Run eval sets and print a pass/fail summary (`--config_file_path` sets the criteria) |
| `adk eval_set` | Create eval sets and add cases to them |
| `adk optimize my_agent --sampler_config_file_path cfg.json` | Improve the root agent's instruction with the GEPA optimiser, scored on your eval sets |
| `adk deploy agent_engine \| cloud_run \| gke \| docker` | Build and deploy the folder; see the [production guide](/google-adk-guide/python/google_adk_production_guide/) |
| `adk migrate` | Database migration helpers for session storage |

Useful options shared by `run`, `web` and `api_server`: `--session_service_uri` (for example `sqlite:///sessions.db`), `--artifact_service_uri`, `--memory_service_uri`, `--otel_to_cloud`, and `--enable_features` / `--disable_features` for experimental feature flags. By default the local commands keep sessions and artifacts in the agent's `.adk/` folder.

`adk web` and `adk api_server` bind to `127.0.0.1` and have no authentication. Do not expose them on a public network; the production guide shows how to serve the same app behind authentication.

## Models

The `model` field takes a model name string or a `BaseLlm` instance. A string is resolved through `LLMRegistry`: `gemini-*` names go to the `Gemini` class, `gpt-*` and `o<N>-*` names to `OpenAILlm` (with the `openai` extra), `claude-*` names to `Claude` on Vertex AI, `gemma-*` names to `Gemma`.

| Role | Model | Use |
|---|---|---|
| General agents and tool calling | `%%model=gemini%%` | Default choice for most agents |
| Routing, classification, high volume | `%%model=gemini-small%%` | Cheapest and fastest |
| Hard reasoning, long documents | `%%model=gemini-flagship%%` | Slower and dearer; use only where it pays off |

`LlmAgent.DEFAULT_MODEL` is `gemini-3.5-flash`, used when an agent and all its ancestors leave `model` empty; `adk create` offers `gemini-3.5-flash` and `gemini-3.8-flash`. The Gemini 2.5 models are no longer offered to new Gemini API projects (only to users who have used them before), so do not copy older examples that name them. Check [Google's models page](https://ai.google.dev/gemini-api/docs/models) for what your project can use, and the [pricing page](https://ai.google.dev/gemini-api/docs/pricing) for current prices.

Generation settings go in `generate_content_config`:

```python
from google.adk.agents import LlmAgent
from google.genai import types

agent = LlmAgent(
    name="careful_writer",
    model="%%model=gemini%%",
    instruction="Write precise, plain answers.",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=2048,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            )
        ],
    ),
)
```

Non-Gemini models (Anthropic, OpenAI, LiteLLM, local models) are covered under [Alternative model providers](/google-adk-guide/python/google_adk_advanced_python/#alternative-model-providers). Retries, fallback models and per-model settings go on a `Gemini(...)` instance; see [Reliability](/google-adk-guide/python/google_adk_production_guide/#reliability).

## Your first agent

An agent does nothing until a `Runner` drives it. `InMemoryRunner` bundles in-memory session, memory and artifact services, which is what you want for scripts and tests:

```python
import asyncio

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types


def get_order_status(order_id: str) -> dict:
    """Look up an order by its ID.

    Args:
        order_id: The order reference, such as "A123".
    """
    return {"order_id": order_id, "status": "shipped", "eta_days": 2}


agent = LlmAgent(
    name="support",
    model="%%model=gemini%%",
    instruction="You are a support agent. Use get_order_status for order questions.",
    tools=[get_order_status],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="support_app")
    session = await runner.session_service.create_session(
        app_name="support_app", user_id="u1"
    )
    message = types.Content(role="user", parts=[types.Part(text="Where is order A123?")])

    async for event in runner.run_async(
        user_id="u1", session_id=session.id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)


asyncio.run(main())
```

Things to know from the start:

- **Create the session first.** `run_async` raises `SessionNotFoundError` for an unknown session. Either call `create_session` as above, or build the runner with `Runner(..., auto_create_session=True)`.
- **A run yields many events**: model text, function calls, function responses, state changes. `event.is_final_response()` marks the one to show the user.
- **`run_debug`** is a shortcut for scripts and notebooks: `await runner.run_debug("Where is order A123?")` creates the session if needed, prints the conversation and returns the events.
- **Synchronous code** can call `runner.run(user_id=..., session_id=..., new_message=...)`, a generator that runs the async loop for you. Do not call it from inside a running event loop.
- **Bound the work.** Pass `run_config=RunConfig(max_llm_calls=20)` (from `google.adk.agents.run_config`) to cap model calls per invocation.

The same agent in an agent folder runs with `adk run` or `adk web` with no runner code at all. [Runner, App and Sessions](/google-adk-guide/python/runner-and-sessions/) covers the runner, `RunConfig` and events in full.

## Tools

A plain Python function becomes a tool. ADK builds the declaration from the signature, type hints and docstring, so write them for the model: say what the tool does, what each argument means, and return a dictionary.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool, ToolContext, google_search


def add_to_basket(sku: str, quantity: int, tool_context: ToolContext) -> dict:
    """Add an item to the customer's basket.

    Args:
        sku: The product SKU.
        quantity: How many to add (1 or more).
    """
    basket = tool_context.state.get("basket", {})
    basket[sku] = basket.get(sku, 0) + quantity
    tool_context.state["basket"] = basket        # persisted with the session
    return {"basket": basket}


researcher = LlmAgent(
    name="researcher",
    model="%%model=gemini%%",
    description="Searches the web and summarises what it finds.",
    instruction="Answer using Google Search. Cite sources.",
    tools=[google_search],
)

shop_assistant = LlmAgent(
    name="shop_assistant",
    model="%%model=gemini%%",
    instruction="Help customers shop. Use the researcher for product questions.",
    tools=[add_to_basket, AgentTool(agent=researcher)],
)
```

- A parameter named `tool_context` is injected by ADK and hidden from the model. It gives access to `state`, artifacts, memory search, credentials and confirmation requests.
- `async def` functions work the same way and are awaited.
- `AgentTool` lets one agent call another like a function and get its answer back; control stays with the caller. This is also how to combine `google_search` with your own function tools.
- Return `{"error": "..."}` from a tool for failures the model should handle. An exception raised in a tool ends the run unless an `on_tool_error_callback` or `ReflectAndRetryToolPlugin` (which turns it into a retry prompt) handles it.

Long-running tools, confirmation prompts (`require_confirmation=`), OpenAPI and Google Cloud toolsets, custom `BaseTool` and `BaseToolset` classes are on the [Tools page](/google-adk-guide/python/tools/).

## Structured output

Set `output_schema` to a Pydantic model and the agent's final reply is constrained to that schema. `output_key` saves the reply in session state for later agents or code:

```python
import asyncio

from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner


class TicketTriage(BaseModel):
    category: str = Field(description="billing, technical or account")
    urgency: int = Field(ge=1, le=5, description="1 = low, 5 = critical")
    summary: str


triage = LlmAgent(
    name="triage",
    model="%%model=gemini-small%%",
    instruction="Classify the support ticket.",
    output_schema=TicketTriage,
    output_key="triage",       # session.state["triage"] holds the validated result
)


async def main():
    runner = InMemoryRunner(agent=triage, app_name="triage_app")
    await runner.run_debug("I was charged twice this month.", quiet=True)
    session = await runner.session_service.get_session(
        app_name="triage_app", user_id="debug_user_id", session_id="debug_session_id"
    )
    print(session.state["triage"])


asyncio.run(main())
```

Tools can be used alongside `output_schema`: the model may call tools during the turn, and the schema applies to the final answer. `input_schema` does the reverse for an agent that is called as a tool or used as a workflow node. More on [the Agents page](/google-adk-guide/python/agents/).

## Multi-agent systems

ADK has three ways to combine agents. Choose by who should be in control:

| Pattern | Mechanism | Control |
|---|---|---|
| Delegation | `sub_agents=[...]` on an `LlmAgent` | The model decides to hand the conversation to a sub-agent (`transfer_to_agent`), which then talks to the user |
| Agent as tool | `tools=[AgentTool(agent=...)]` | The calling agent stays in charge and receives the result |
| Workflow | `Workflow(edges=[...])` | Your graph decides the order; agents and functions are nodes |

Delegation, with each specialist's `description` telling the coordinator when to use it:

```python
from google.adk.agents import LlmAgent

billing = LlmAgent(
    name="billing",
    model="%%model=gemini%%",
    description="Handles invoices, refunds and payment problems.",
    instruction="Resolve billing questions. Ask for the invoice number.",
)

technical = LlmAgent(
    name="technical",
    model="%%model=gemini%%",
    description="Handles bugs, errors and how-to questions about the product.",
    instruction="Diagnose technical problems step by step.",
)

coordinator = LlmAgent(
    name="coordinator",
    model="%%model=gemini-small%%",
    instruction="Work out what the customer needs and transfer to the right specialist.",
    sub_agents=[billing, technical],
)
```

- Each agent can have only one parent; ADK sets `parent_agent` when you pass `sub_agents` to the constructor.
- After a transfer the sub-agent answers subsequent turns until it transfers back. `disallow_transfer_to_parent` and `disallow_transfer_to_peers` restrict where it can go.
- `mode="single_turn"` or `mode="task"` turns a sub-agent into one that completes a single job and returns control, which is usually what you want for specialists inside a larger flow (see [LlmAgent modes](/google-adk-guide/python/agents/#llmagent-modes)).

## Workflows

For fixed pipelines, fan-out and joins, routing and loops, use `Workflow`. Agents, functions and tools appear directly in `edges`; a tuple is a chain, a nested tuple fans out, a dict routes:

```python
import asyncio

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.workflow import START, JoinNode, Workflow, node

security = LlmAgent(
    name="security",
    model="%%model=gemini%%",
    instruction="List security issues in the code. Bullet points only.",
    mode="single_turn",
)
style = LlmAgent(
    name="style",
    model="%%model=gemini-small%%",
    instruction="List readability issues in the code. Bullet points only.",
    mode="single_turn",
)


@node
def report(node_input: dict) -> str:
    # A JoinNode passes {branch_name: output} to the next node.
    return "\n\n".join(f"## {name}\n{text}" for name, text in node_input.items())


review = Workflow(
    name="code_review",
    edges=[(START, (security, style), JoinNode(name="join"), report)],
)


async def main():
    runner = InMemoryRunner(app=App(name="review_app", root_agent=review))
    events = await runner.run_debug("def f(q): return db.execute(f'SELECT {q}')", quiet=True)
    print(events[-1].output)   # function nodes put their result in event.output


asyncio.run(main())
```

- A function node receives the previous node's output in a parameter named `node_input`; other parameters are filled from session state by name.
- Run a `Workflow` as the root of an `App`, or give it an `input_schema` and pass it to an `LlmAgent` in `tools=[...]`. It cannot be an `LlmAgent` sub-agent.
- `@node(retry_config=RetryConfig(...), timeout=30.0)` adds retries and timeouts per node; `require_confirmation=True` (2.12.0) pauses for human approval before a node runs.

Coming from the deprecated agents: `SequentialAgent([a, b, c])` becomes `edges=[(START, a, b, c)]`, `ParallelAgent([a, b])` becomes a fan-out into a `JoinNode`, and a `LoopAgent` becomes an edge back to an earlier node with a route that exits. The [Workflows page](/google-adk-guide/python/workflows/) has all of these, plus dynamic nodes, state schemas and human-in-the-loop pauses.

## State, sessions and memory

**Session state** is a dictionary saved with the session. Key prefixes set its scope:

| Prefix | Scope |
|---|---|
| none | This session |
| `user:` | Every session of this user in this app |
| `app:` | Every user and session of this app |
| `temp:` | The current invocation only; never stored |

Write state from tools and callbacks (`tool_context.state["key"] = value`), with `output_key`, or with `state_delta=` on `run_async`. Never put secrets in state: it is stored with the session and can be shown to the model through instructions.

**Session services** decide where sessions live: `InMemorySessionService` for tests, `DatabaseSessionService` (SQLAlchemy URL; `pip install "google-adk[db]"`) or `SqliteSessionService` for your own database, `VertexAiSessionService` for Agent Platform Sessions.

**Memory** is searchable knowledge across sessions. Add a finished session to the memory service, then give agents a memory tool:

```python
import asyncio

from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory

memory = InMemoryMemoryService()
sessions = InMemorySessionService()

agent = LlmAgent(
    name="assistant",
    model="%%model=gemini%%",
    instruction="Use load_memory when the user refers to earlier conversations.",
    tools=[load_memory],
)
runner = Runner(
    app_name="mem_app",
    agent=agent,
    session_service=sessions,
    memory_service=memory,
    auto_create_session=True,
)


async def main():
    await runner.run_debug("My favourite colour is green.", user_id="u1", session_id="s1", quiet=True)
    done = await sessions.get_session(app_name="mem_app", user_id="u1", session_id="s1")
    await memory.add_session_to_memory(done)

    found = await memory.search_memory(app_name="mem_app", user_id="u1", query="favourite colour")
    print(len(found.memories), "memories found")


asyncio.run(main())
```

`load_memory` lets the model search when it decides to; `preload_memory` searches on every turn and adds the results to the prompt. In production, `VertexAiMemoryBankService` (Memory Bank) extracts and consolidates memories for you. **Artifacts** are versioned files (images, PDFs, reports) saved with `tool_context.save_artifact(filename, part)` and stored by an artifact service (in-memory, local files or Cloud Storage). See [Runner, App and Sessions](/google-adk-guide/python/runner-and-sessions/) and [Memory and Artifacts](/google-adk-guide/python/memory-and-artifacts/).

## Instructions and context

`instruction` is a template. `{key}` inserts a state value, `{key?}` inserts it or nothing if it is missing, and `{artifact.filename}` inserts a text artifact:

```python
from google.adk.agents import LlmAgent

writer = LlmAgent(
    name="writer",
    model="%%model=gemini%%",
    static_instruction="You write product descriptions for Example Ltd. House style: British English, no superlatives.",
    instruction="Write for a {user:audience?} audience. Current draft, if any: {draft?}",
    output_key="draft",
)
```

- `static_instruction` never changes, so it can be cached; `instruction` changes per turn. Keeping them apart helps both implicit and explicit context caching.
- An instruction can also be a function `(ReadonlyContext) -> str` when the template language is not enough; `inject_session_state(template, ctx, use_jinja2=True)` (from `google.adk.flows.llm_flows.prompt`, 2.7.0) renders Jinja2 templates with conditionals and loops inside such a function. Since 2.10.0, `${var}` and `\{var}` are left alone, so you can write literal braces.
- `include_contents="none"` sends a sub-agent only its instruction and the current input, not the whole conversation.
- For long sessions, `App(events_compaction_config=EventsCompactionConfig(...))` summarises old events, and `App(context_cache_config=ContextCacheConfig(...))` caches the stable prefix. Both are covered in [Runner, App and Sessions](/google-adk-guide/python/runner-and-sessions/#context-caching-contextcacheconfig) and in the production guide's [cost section](/google-adk-guide/python/google_adk_production_guide/#cost-and-context-controls).

## Callbacks and plugins

**Callbacks** attach to one agent and run before or after its agent, model and tool steps. Returning a value instead of `None` replaces the step:

```python
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

BLOCKED = ("password", "credit card number")


def screen_input(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
    last = llm_request.contents[-1] if llm_request.contents else None
    text = " ".join(p.text or "" for p in (last.parts or [])) if last else ""
    if any(term in text.lower() for term in BLOCKED):
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="I can't help with that.")])
        )
    return None   # carry on to the model


agent = LlmAgent(
    name="guarded",
    model="%%model=gemini%%",
    instruction="Help with account questions.",
    before_model_callback=screen_input,
)
```

**Plugins** (`BasePlugin` subclasses) do the same for every agent, model and tool call in an app. Register them on the `App` or `Runner`. `BasePlugin` has 14 hooks, including run-level ones such as `on_user_message_callback`, `on_event_callback` and `after_run_callback`. Built-in plugins cover logging, tool retries, global instructions, context filtering, saving uploads as artifacts, BigQuery analytics and Model Armor screening. All of it is on [Callbacks and Plugins](/google-adk-guide/python/callbacks-and-plugins/).

## MCP and A2A

**MCP.** `McpToolset` connects to a Model Context Protocol server over stdio, SSE or streamable HTTP and offers its tools to an agent (`pip install "google-adk[mcp]"`):

```python
from mcp import StdioServerParameters

from google.adk.agents import LlmAgent
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams

files = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/srv/docs"],
        ),
    ),
    tool_filter=["read_file", "list_directory"],
)

agent = LlmAgent(
    name="docs_agent",
    model="%%model=gemini%%",
    instruction="Answer questions from the documents in /srv/docs.",
    tools=[files],
)
```

In the other direction, `to_mcp_server` exposes an ADK agent as an MCP server.

**A2A.** The Agent2Agent protocol connects agents across processes and frameworks (`pip install "google-adk[a2a]"`). `to_a2a(agent)` turns an agent into an A2A server (a Starlette app with an agent card), and `RemoteA2aAgent(name=..., agent_card=URL)` wraps a remote A2A agent so it can be used as a sub-agent. `adk api_server --a2a` and `adk deploy cloud_run --a2a` serve the A2A endpoint alongside the normal API. Details, authentication and task mode are on [MCP and A2A](/google-adk-guide/python/mcp-and-a2a/).

## Gemini features

**Grounding with Google Search** and **URL context** are model-side tools: pass `google_search` or `url_context` from `google.adk.tools`. Since 2.12.0 an agent with sub-agents can use them on Gemini 3 models; to mix them with your own function tools in one agent, wrap the searching agent in `AgentTool` as shown under [Tools](#tools).

**Code execution** runs model-written Python in Gemini's sandbox:

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor

analyst = LlmAgent(
    name="analyst",
    model="%%model=gemini%%",
    instruction="Use Python to calculate answers. Show the result, not the code.",
    code_executor=BuiltInCodeExecutor(),
)
```

Other executors (`VertexAiCodeExecutor`, container, GKE and Agent Engine sandboxes) are listed under [Code executors](/google-adk-guide/python/agents/#code-executors).

**Multimodal input.** Send images, audio, video or PDFs as `Part`s in the user message:

```python
from google.genai import types

with open("receipt.png", "rb") as f:
    image_bytes = f.read()

message = types.Content(
    role="user",
    parts=[
        types.Part(text="What is the total on this receipt?"),
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
    ],
)
# runner.run_async(user_id=..., session_id=..., new_message=message)
```

**Thinking.** Gemini 3 models take `thinking_config=types.ThinkingConfig(thinking_level=...)` in `generate_content_config`, or through `BuiltInPlanner`. Thoughts can be returned with `include_thoughts=True`; see [Planners](/google-adk-guide/python/google_adk_advanced_python/#planners).

**Safety settings** go in `generate_content_config` (see [Models](#models)); Model Armor screening is a plugin (see the [production guide](/google-adk-guide/python/google_adk_production_guide/#guardrails)).

## Streaming and live sessions

There are two kinds of streaming.

**Token streaming** for text: pass `RunConfig(streaming_mode=StreamingMode.SSE)` to `run_async`, and events with `event.partial == True` carry chunks of the reply as they arrive. The API server's `/run_sse` endpoint uses this.

**Live (bidirectional) sessions** for voice and video: `runner.run_live()` keeps a connection to a Gemini Live model open. You push input into a `LiveRequestQueue` and read events as the model speaks:

```python
import asyncio

from google.adk.agents import LlmAgent, LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import InMemoryRunner
from google.genai import types

voice_agent = LlmAgent(
    name="voice",
    model=LlmAgent.DEFAULT_LIVE_MODEL,   # a Gemini Live model; check which ones your project can use
    instruction="You are a friendly voice assistant. Keep answers short.",
)


async def converse(runner: InMemoryRunner, user_id: str, session_id: str) -> None:
    queue = LiveRequestQueue()
    run_config = RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    async def send() -> None:
        queue.send_content(types.Content(role="user", parts=[types.Part(text="Hello!")]))
        # For audio: queue.send_realtime(types.Blob(mime_type="audio/pcm;rate=16000", data=chunk))
        await asyncio.sleep(10)
        queue.close()

    sender = asyncio.create_task(send())
    async for event in runner.run_live(
        user_id=user_id, session_id=session_id, live_request_queue=queue, run_config=run_config
    ):
        if event.output_transcription and event.output_transcription.text:
            print(event.output_transcription.text, end="", flush=True)
    await sender
```

- `send_content` sends a text turn, `send_realtime` streams audio or video chunks, `send_activity_start` / `send_activity_end` mark speech when you do your own voice-activity detection, and `close` ends the session.
- Tools work in live sessions; a tool with a parameter annotated `LiveRequestQueue` receives a stream of its own.
- `LlmAgent.DEFAULT_LIVE_MODEL` is ADK's built-in default live model (`gemini-live-2.5-flash-native-audio` in this release); live model names change often, so set the one your project has access to.
- `adk web` and `adk api_server` expose this as the `/run_live` WebSocket endpoint.

`RunConfig` options for live sessions (speech, transcription, session resumption, context window compression) are in [Runner, App and Sessions](/google-adk-guide/python/runner-and-sessions/#runconfig).

## Evaluation and optimisation

ADK evaluates an agent against **eval sets**: recorded conversations with the expected final responses and, optionally, the expected tool calls. Create them in the Dev UI from real sessions, or write the JSON by hand, then run them from the command line or from pytest:

```bash
pip install "google-adk[eval]"
adk eval my_agent my_agent/tests/smoke.evalset.json --config_file_path my_agent/test_config.json
```

with `test_config.json` setting the pass thresholds, for example `{"criteria": {"tool_trajectory_avg_score": 1.0, "response_match_score": 0.8}}`. Metrics range from exact tool-trajectory matching and ROUGE through LLM-judged response quality, rubrics, hallucination and safety checks to multi-turn user simulation. See [Evaluation](/google-adk-guide/python/evaluation/).

`adk optimize` (and the `google.adk.optimization` module) rewrites the root agent's instruction with the GEPA optimiser, scoring each candidate on your eval sets; see [Optimising the instruction with GEPA](/google-adk-guide/python/evaluation/#optimising-the-instruction-with-gepa). Treat its output as a proposal to review, not a change to ship unread.

## Deployment

`adk deploy` builds and deploys an agent folder to Vertex AI Agent Engine, Cloud Run or GKE; the Agent Platform SDK's `AdkApp` does the same from Python. The [production guide](/google-adk-guide/python/google_adk_production_guide/) covers the options, persistent sessions, authentication, telemetry, cost controls, guardrails and CI.

## Import paths worth knowing

Most classes have a short import (`from google.adk.agents import LlmAgent`, `from google.adk.tools import FunctionTool`). Some useful ones are only in submodules:

| Class or function | Import from |
|---|---|
| `App`, `ResumabilityConfig` | `google.adk.apps` |
| `EventsCompactionConfig` | `google.adk.apps.app` |
| `ContextCacheConfig` | `google.adk.agents.context_cache_config` |
| `RunConfig`, `StreamingMode` | `google.adk.agents.run_config` |
| `Workflow`, `node`, `JoinNode`, `RetryConfig`, `START` | `google.adk.workflow` |
| `to_a2a` | `google.adk.a2a.utils.agent_to_a2a` |
| `RemoteA2aAgent` | `google.adk.agents.remote_a2a_agent` |
| `StdioConnectionParams`, `SseConnectionParams`, `StreamableHTTPConnectionParams` | `google.adk.tools.mcp_tool` |
| `AnthropicLlm`, `Claude` | `google.adk.models.anthropic_llm` |
| `LiteLlm` | `google.adk.models.lite_llm` |
| `OpenAILlm`, `OpenAIResponsesLlm` | `google.adk.integrations.openai` (since 2.10.0; `google.adk.labs.openai` re-exports them) |
| `FallbackModel` | `google.adk.models` |
| `FirestoreSessionService`, BigQuery, GCS and other Google Cloud integrations | `google.adk.integrations.*` |
| `ModelArmorPlugin`, `ModelArmorConfig` | `google.adk.integrations.model_armor` |
| `get_fast_api_app` | `google.adk.cli.fast_api` |

Modules and names starting with an underscore are internal and can change in any release; if a class is only importable from such a path, it is not part of the supported API.

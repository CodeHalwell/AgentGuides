---
title: "Microsoft Agent Framework (Python) — Comprehensive Guide"
description: "The core of Microsoft Agent Framework for Python in one page: installation, configuration, agents, tools, structured output, streaming, sessions, graph workflows, orchestration, MCP, vector stores and prompt-injection defence, with links to the topic pages."
framework: "microsoft-agent-framework"
language: "python"
sidebar:
  label: "Comprehensive guide"
---

**Version:** `agent-framework` %%version=agent-framework%% (`agent-framework-core` is released in lockstep), checked %%versions-checked%%. Python 3.10+.

This page covers the parts of the framework most applications use, in the order you meet them. Each section ends with a link to the topic page that goes deeper. The [API map](#api-map) at the end lists where every major class is documented.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install agent-framework        # meta package: core plus the provider and integration packages
# pin for reproducible builds: pip install agent-framework==%%version=agent-framework%%
```

The meta package is convenient for development. For services, install only what you use:

```bash
pip install agent-framework-core            # Agent, tools, sessions, workflows, FIDES security, evaluation
pip install agent-framework-openai          # OpenAIChatClient (OpenAI and Azure OpenAI)
pip install agent-framework-foundry         # FoundryChatClient (Microsoft Foundry projects)
pip install agent-framework-orchestrations  # Sequential / Concurrent / Handoff / GroupChat / Magentic builders
pip install azure-identity                  # DefaultAzureCredential, managed identity
pip install "graphviz>=0.20.0"              # optional: WorkflowViz SVG/PNG/PDF export (also needs the dot binary)
```

There are no `[security]` or `[evals]` extras: FIDES and evaluation ship in `agent-framework-core`. Provider packages that are still in the `1.0.0b` line (Anthropic, Bedrock, Ollama, Gemini, Mistral, A2A and others) need `--pre` when installed on their own. The legacy `agent-framework-azure-ai` package stopped at 1.0.0rc6; use `agent-framework-foundry` instead.

## Configuration

### Settings with `load_settings` and `SecretString`

`load_settings` reads a `TypedDict` from environment variables, an optional `.env` file and keyword overrides. Fields typed `SecretString` are wrapped so they print as `**********`.

```text
# .env
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
AZURE_OPENAI_DEPLOYMENT_NAME="%%model=openai%%"
```

```python
# config.py
from typing import TypedDict
from agent_framework import SecretString, load_settings


class AzureSettings(TypedDict, total=False):
    endpoint: str | None
    api_key: SecretString | None   # wrapped automatically; prints as **********
    deployment_name: str | None
    max_tokens: int | None         # int / float / bool fields are coerced from the env string


settings = load_settings(
    AzureSettings,
    env_prefix="AZURE_OPENAI_",    # reads AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, ...
    required_fields=["endpoint", "deployment_name"],
)

AZURE_OPENAI_ENDPOINT = settings["endpoint"]
AZURE_OPENAI_DEPLOYMENT = settings["deployment_name"]
# Unwrap a secret only at the SDK boundary: settings["api_key"].get_secret_value()
```

Resolution order, highest first: keyword overrides, the `.env` file (only when `env_file_path=` is given; the file must then exist), environment variables, then `TypedDict` defaults. `required_fields` also accepts tuples meaning "exactly one of these", for example `("api_key", "ad_token")`. A failed check raises `agent_framework.exceptions.SettingNotFoundError`.

`SecretString` is not a `str` subclass: `str()`, `repr()` and f-strings are masked, `"Bearer " + secret` gives `"Bearer **********"`, and `json.dumps` or `str.join` raise `TypeError` instead of leaking the value.

Chat clients also read their own environment variables (`OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `FOUNDRY_PROJECT_ENDPOINT` and so on), so many applications need no settings code at all.

### Credentials

In Azure, prefer Microsoft Entra ID over API keys. `DefaultAzureCredential` uses managed identity in Azure, and environment variables or your `az login` session locally:

```python
from azure.identity.aio import DefaultAzureCredential

credential = DefaultAzureCredential()
```

The roles each service needs are listed on the [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/).

## Your first agent

```python
# main.py
import asyncio
from azure.identity.aio import DefaultAzureCredential

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from config import AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_ENDPOINT


async def main() -> None:
    async with DefaultAzureCredential() as credential:
        # Azure OpenAI is served by OpenAIChatClient: pass azure_endpoint= and a credential.
        client = OpenAIChatClient(
            AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            credential=credential,
        )
        agent = Agent(client=client, instructions="You are a helpful assistant for Python developers.")

        response = await agent.run("What are decorators in Python?")
        print(response.text)


if __name__ == "__main__":
    asyncio.run(main())
```

Swap the client to change provider: `FoundryChatClient` (`agent_framework.foundry`), `AnthropicClient` (`agent_framework.anthropic`), `OllamaChatClient`, `BedrockChatClient`, `GeminiChatClient` and others. Constructors and environment variables for each are on the [model providers page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_model_providers/).

## Agents

### `Agent`, `RawAgent` and `BaseAgent`

| Class | Use |
|---|---|
| `Agent` | The default. Chat-client agent with the middleware and telemetry layers. |
| `RawAgent` | Same constructor and `run()` signature, without the middleware and telemetry layers. It accepts `middleware=` but never runs it. For tight inner loops and tests. |
| `BaseAgent` | Abstract base for agents that are not backed by a chat client (see [Advanced patterns](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_advanced/#custom-agent--baseagent-and-rawagent)). |

A call without a session is stateless. For a conversation, create a session and pass it to every call; the session's history provider (in memory by default) accumulates the turns:

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="You are a friendly assistant.")

    session = agent.create_session()
    await agent.run("My name is Ada.", session=session)
    response = await agent.run("What is my name?", session=session)
    print(response.text)


asyncio.run(main())
```

Agents are cheap to construct; build one per role and reuse it across requests. Sessions are per conversation. The chat client owns the HTTP connection pool, so close it at shutdown (`async with` where the client supports it).

### Constructor reference

`Agent(client, instructions=None, *, ...)`. Everything after `instructions` is keyword-only.

| Parameter | Default | Description |
|---|---|---|
| `client` | required | Chat client (`OpenAIChatClient`, `FoundryChatClient`, ...). |
| `instructions` | `None` | System prompt. |
| `id` | generated | Stable identifier, used in workflow graphs and telemetry. |
| `name` | `None` | Shown in multi-agent event streams (`update.author_name`). |
| `description` | `None` | Shown to an orchestrating model (for example by `HandoffBuilder`) to decide when to route here. |
| `tools` | `None` | Default tools: `@tool` functions, plain callables, MCP tools or hosted tools. |
| `default_options` | `None` | Default chat options (`model`, `temperature`, `max_tokens`, `response_format`, ...). |
| `context_providers` | `None` | Ordered `ContextProvider`s (history, memory, skills, harness providers). |
| `middleware` | `None` | List of agent, chat and function middleware. |
| `compaction_strategy` | `None` | Default history compaction strategy. |
| `tokenizer` | `None` | Tokenizer used by compaction; falls back to a character estimate. |
| `require_per_service_call_history_persistence` | `False` | Persist history after every model call inside a tool loop, not only at the end of the run. |
| `additional_properties` | `None` | Free-form mapping that middleware and providers can read. |

```python
from agent_framework import Agent, SlidingWindowStrategy, tool
from agent_framework.openai import OpenAIChatClient


@tool
def search_codebase(query: str) -> str:
    """Search the repository."""
    return "no matches"


agent = Agent(
    client=OpenAIChatClient(model="%%model=openai%%"),
    instructions="You are a senior code reviewer.",
    id="code-reviewer-1",
    name="CodeReviewer",
    description="Reviews code diffs and flags issues.",
    tools=[search_codebase],
    default_options={"temperature": 0.2, "max_tokens": 2048},
    compaction_strategy=SlidingWindowStrategy(keep_last_groups=20),
    require_per_service_call_history_persistence=True,
    additional_properties={"team": "platform"},
)
```

### `run()` reference

`agent.run(messages=None, *, stream=False, session=None, middleware=None, tools=None, options=None, compaction_strategy=None, tokenizer=None, function_invocation_kwargs=None, client_kwargs=None)`

| Parameter | Description |
|---|---|
| `messages` | A string (treated as a user message), a `Message`, or a list of them. |
| `stream` | `True` returns a `ResponseStream` instead of an awaitable. |
| `session` | `AgentSession` carrying history. Omit for a stateless call. |
| `middleware` | Extra middleware for this run, added to the agent's own. |
| `tools` | Extra tools for this run, merged with the agent's default tools. |
| `options` | Chat options for this run, merged field by field over `default_options`. |
| `compaction_strategy`, `tokenizer` | Per-run overrides. |
| `function_invocation_kwargs` | Values passed to tools (and MCP `header_provider`s) that accept `**kwargs`: tenant id, auth token, trace id. Not shown to the model. |
| `client_kwargs` | Provider-specific keyword arguments passed to the chat client call. |

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

agent = Agent(client=OpenAIChatClient(), instructions="You are a helpful assistant.",
              default_options={"temperature": 0.7})


async def main() -> None:
    label = await agent.run(
        "Classify the sentiment: 'The product is amazing!'",
        options={"model": "%%model=openai-small%%", "temperature": 0.0},
        function_invocation_kwargs={"tenant_id": "acme"},
    )
    print(label.text)


asyncio.run(main())
```

## Tools

A tool is a Python function. `@tool` (from `agent_framework`) builds a `FunctionTool` with a JSON schema from the type hints and docstring; plain callables passed in `tools=` are wrapped the same way.

```python
from datetime import datetime
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient


@tool(description="Get the current time in a timezone.")
def get_current_time(timezone: Annotated[str, "IANA timezone name, e.g. 'America/New_York'."]) -> str:
    try:
        return datetime.now(ZoneInfo(timezone)).strftime("%H:%M:%S")
    except ZoneInfoNotFoundError:
        return "Unknown timezone."


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You can tell the time.",
    tools=[get_current_time],
)
```

`@tool(approval_mode="always_require")` pauses for human approval before the call runs (see [Human in the loop](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_hitl/)). Schemas, `max_invocations`, result parsers, declaration-only tools and the `Supports*` protocols for provider-hosted tools (web search, file search, code interpreter, hosted MCP) are on the [tools page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_tools/). For Azure AI Search retrieval, see the [recipes page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_recipes/).

## Structured output

Pass a Pydantic model as `response_format` in the chat options. `response.value` holds the parsed instance.

```python
from pydantic import BaseModel, Field

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


class UserProfile(BaseModel):
    name: str = Field(description="The user's full name.")
    age: int
    interests: list[str]


async def extract_profile(text: str) -> UserProfile:
    agent = Agent(client=OpenAIChatClient(), instructions="Extract the user's profile from the text.")
    response = await agent.run(text, options={"response_format": UserProfile})
    return response.value
```

Set `default_options={"response_format": UserProfile}` on the agent to apply it to every call. With `stream=True`, the parsed value is available on the final response from `await stream.get_final_response()`.

## Streaming

`agent.run(..., stream=True)` returns a `ResponseStream[AgentResponseUpdate, AgentResponse]`. Iterate it for updates; afterwards `await stream.get_final_response()` returns the assembled `AgentResponse`.

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="You are a helpful assistant.")

    stream = agent.run("Explain backpressure in three short paragraphs.", stream=True)
    async for update in stream:
        if update.text:
            print(update.text, end="", flush=True)
    print()

    final = await stream.get_final_response()
    print(final.usage_details)


asyncio.run(main())
```

Each `AgentResponseUpdate` carries `contents`, `role`, `author_name`, `agent_id`, `response_id`, `message_id`, `created_at`, `finish_reason` and `continuation_token`. In multi-agent runs, `author_name` tells you which agent produced the chunk. Approval requests arrive as content items, collected on `update.user_input_requests`; there is no way to answer them mid-stream. Let the run finish, then reply on the same session with the approval responses, as shown on the [HITL page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_hitl/).

### Persisting and replaying updates

Updates serialise with `to_json()` / `from_json()` (and `to_dict()` / `from_dict()`), which is useful for queues, websockets and test fixtures. `AgentResponse.from_updates()` rebuilds a full response from a list of them:

```python
from pydantic import BaseModel
from agent_framework import AgentResponse, AgentResponseUpdate, Content


class ReleasePlan(BaseModel):
    version: str
    date: str


chunks = [
    AgentResponseUpdate(contents=[Content.from_text('{"version": "2.1", ')], role="assistant").to_json(),
    AgentResponseUpdate(contents=[Content.from_text('"date": "2026-11-02"}')], role="assistant").to_json(),
]

updates = [AgentResponseUpdate.from_json(line) for line in chunks]
final = AgentResponse.from_updates(updates, output_format_type=ReleasePlan)
print(final.text)
print(final.value.version)        # parsed on first access
```

`AgentResponse.from_update_generator(stream)` does the same for an async iterator.

## Sessions and history

`AgentSession` is a `session_id` plus a `state` dict that context providers read and write. History is stored by a `HistoryProvider`. The agent adds an `InMemoryHistoryProvider` when you don't configure one, which keeps messages in `session.state` for the life of the process.

| Need | Use |
|---|---|
| History survives restarts on one host | `FileHistoryProvider(storage_path)` (experimental, `ExperimentalFeature.FILE_HISTORY`): one JSON Lines file per `session_id`, path-traversal checks, optional `dumps=` / `loads=` for encryption |
| Several processes share history | Redis (`agent-framework-redis`) or Cosmos DB (`agent-framework-azure-cosmos`) history providers |
| Your own store (Postgres, S3, ...) | Subclass `HistoryProvider` and implement `get_messages` and `save_messages` |
| Move a whole session between workers | `session.to_dict()` / `AgentSession.from_dict()`, or `FileSessionStore` |
| Provider keeps the history (OpenAI Responses, Foundry) | Persist only `session_id` and `service_session_id` |

All of these, plus write-only audit copies and `VectorStoreHistoryProvider`, are covered with examples on the [sessions page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sessions/).

## Compaction

Compaction decides what the model sees on each turn while keeping the stored history intact. A `CompactionProvider` runs one strategy before the model call and another on the stored history afterwards:

```python
from agent_framework import (
    Agent,
    CompactionProvider,
    InMemoryHistoryProvider,
    SlidingWindowStrategy,
    ToolResultCompactionStrategy,
)
from agent_framework.openai import OpenAIChatClient

history = InMemoryHistoryProvider()
compaction = CompactionProvider(
    before_strategy=SlidingWindowStrategy(keep_last_groups=20),
    after_strategy=ToolResultCompactionStrategy(keep_last_tool_call_groups=1),
    history_source_id=history.source_id,
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a research assistant.",
    context_providers=[history, compaction],
)
```

The built-in strategies are `TruncationStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`, `ToolResultCompactionStrategy`, `SummarizationStrategy` (model-driven), `TokenBudgetComposedStrategy` and `ContextWindowCompactionStrategy`. Trade-offs and custom strategies are on the [compaction page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_compaction/).

## Middleware

Middleware wraps the agent run (`AgentMiddleware`), each model call in the tool loop (`ChatMiddleware`) or each tool call (`FunctionMiddleware`). Call `call_next()` to continue; raise `MiddlewareTermination` to stop.

```python
from agent_framework import Agent, AgentContext, AgentMiddleware, MiddlewareTermination
from agent_framework.openai import OpenAIChatClient


class BudgetGuard(AgentMiddleware):
    def __init__(self, max_runs: int) -> None:
        self.remaining = max_runs

    async def process(self, context: AgentContext, call_next) -> None:
        if self.remaining <= 0:
            raise MiddlewareTermination("budget exhausted")
        self.remaining -= 1
        await call_next()


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    middleware=[BudgetGuard(max_runs=20)],
)
```

The decorator forms (`@agent_middleware`, `@chat_middleware`, `@function_middleware`), caching, retries, redaction and streaming hooks are on the [middleware page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_middleware/).

## Workflows

`WorkflowBuilder` wires executors (deterministic steps) and agents into a directed graph. The runner works in supersteps: every pending message is delivered to its target in parallel, the targets' outgoing messages are collected, and the cycle repeats until no messages remain or `max_iterations` (default 100) is reached. `workflow.run(message)` returns a `WorkflowRunResult`; `workflow.run(message, stream=True)` returns a stream of `WorkflowEvent`s.

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()
    researcher = Agent(client=client, name="researcher", instructions="Bullet-point findings.")
    writer = Agent(client=client, name="writer", instructions="One-paragraph summary.")

    workflow = WorkflowBuilder(start_executor=researcher, name="research").add_edge(researcher, writer).build()

    result = await workflow.run("Quantum sensors")
    for output in result.get_outputs():
        print(output)


asyncio.run(main())
```

Agents passed to the builder are wrapped in an `AgentExecutor` automatically (once per agent). Wrap them yourself when you need a different `id` or another `context_mode`:

- `"full"` (default): the downstream agent receives the whole conversation so far.
- `"last_agent"`: only the previous agent's response, which keeps token use flat on long chains.
- `"custom"`: pass `context_filter=` to shape the messages yourself.

```python
from agent_framework import Agent, AgentExecutor
from agent_framework.openai import OpenAIChatClient

researcher = Agent(client=OpenAIChatClient(), name="researcher", instructions="Bullet-point findings.")
research_node = AgentExecutor(researcher, id="research", context_mode="last_agent")
```

Graph and functional workflows (`@workflow` / `@step`) share the same runtime. The [functional workflows page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_functional_workflows/) covers the decorator style and when to prefer it.

### Custom executors

A function executor is the simplest deterministic step. The `WorkflowContext` type parameters declare what the executor sends downstream (first) and what it yields as workflow output (second):

```python
from agent_framework import AgentExecutorResponse, WorkflowContext, executor


@executor(id="upper_case")
async def upper_case(
    response: AgentExecutorResponse,
    ctx: WorkflowContext[AgentExecutorResponse, str],
) -> None:
    upper_text = response.agent_response.text.upper()
    await ctx.send_message(response.with_text(upper_text))   # keeps the conversation for the next agent
    await ctx.yield_output(upper_text)
```

If a custom executor sent a plain `str` to the next agent, that agent would see only the string; `AgentExecutorResponse.with_text()` keeps the full conversation attached.

For several input types or state that lasts across invocations, subclass `Executor` and mark methods with `@handler`. The runtime dispatches on the message type:

```python
from agent_framework import Executor, WorkflowContext, handler


class Counter(Executor):
    def __init__(self) -> None:
        super().__init__(id="counter")
        self._count = 0

    @handler
    async def tick(self, _: str, ctx: WorkflowContext[str]) -> None:
        self._count += 1
        await ctx.send_message(f"count={self._count}")

    @handler
    async def reset(self, _: int, ctx: WorkflowContext[str]) -> None:
        self._count = 0
        await ctx.send_message("reset")
```

When annotations are inconvenient (forward references, dynamically built executors), give the types to the decorator instead: `@handler(input=str | int, output=bool, workflow_output=str)` or `@executor(id=..., input=..., output=...)`. Once any explicit type is given, annotations are ignored, so give all of them.

An executor that only yields output uses `Never` (from `typing`, or `typing_extensions` on Python 3.10) as the first type parameter: `WorkflowContext[Never, str]`.

### Routing: fan-out, fan-in, switch-case, multi-selection

**Fan-out and fan-in.** `add_fan_out_edges(source, targets)` sends each message to every target in the same superstep. `add_fan_in_edges(sources, target)` delivers the sources' messages to the target together, as one list, so the target's input type is `list[T]`.

```python
import asyncio
from typing_extensions import Never

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler


class Splitter(Executor):
    @handler
    async def split(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(text)


class Worker(Executor):
    def __init__(self, id: str, suffix: str) -> None:
        super().__init__(id=id)
        self.suffix = suffix

    @handler
    async def work(self, text: str, ctx: WorkflowContext[str, str]) -> None:
        await ctx.send_message(f"{text}-{self.suffix}")
        await ctx.yield_output(f"{self.id} done")      # progress note, also a workflow output


class Aggregator(Executor):
    @handler
    async def aggregate(self, results: list[str], ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(" | ".join(sorted(results)))


splitter = Splitter(id="splitter")
workers = [Worker("a", "A"), Worker("b", "B"), Worker("c", "C")]
aggregator = Aggregator(id="aggregator")

workflow = (
    WorkflowBuilder(start_executor=splitter)
    .add_fan_out_edges(splitter, workers)
    .add_fan_in_edges(workers, aggregator)
    .build()
)


async def main() -> None:
    result = await workflow.run("job")
    print(result.get_outputs())     # ['a done', 'b done', 'c done', 'job-A | job-B | job-C']


asyncio.run(main())
```

**Switch-case.** `add_switch_case_edge_group(source, [Case(...), ..., Default(...)])` evaluates the conditions in order and sends the message to the first match. Exactly one `Default` is required.

```python
import asyncio
from dataclasses import dataclass
from typing_extensions import Never

from agent_framework import Case, Default, Executor, WorkflowBuilder, WorkflowContext, executor, handler


@dataclass
class ScoredText:
    text: str
    score: int


class Scorer(Executor):
    @handler
    async def score(self, text: str, ctx: WorkflowContext[ScoredText]) -> None:
        await ctx.send_message(ScoredText(text=text, score=len(text.split())))


@executor(id="long-handler")
async def long_handler(payload: ScoredText, ctx: WorkflowContext[Never, str]) -> None:
    await ctx.yield_output(f"[LONG] {payload.text[:40]}...")


@executor(id="short-handler")
async def short_handler(payload: ScoredText, ctx: WorkflowContext[Never, str]) -> None:
    await ctx.yield_output(f"[SHORT] {payload.text}")


scorer = Scorer(id="scorer")
workflow = (
    WorkflowBuilder(start_executor=scorer)
    .add_switch_case_edge_group(
        scorer,
        [
            Case(condition=lambda p: p.score > 50, target=long_handler),
            Default(target=short_handler),
        ],
    )
    .build()
)


async def main() -> None:
    print((await workflow.run("Hello world")).get_outputs())          # ['[SHORT] Hello world']
    print((await workflow.run(" ".join(["word"] * 60))).get_outputs())  # ['[LONG] word word ...']


asyncio.run(main())
```

Targets must accept the type the source sends. An `Agent` target accepts `str`, `Message` or a list of these, so routing a dataclass straight to an agent fails at `build()` with `TypeCompatibilityError`. Put a small executor in between that turns the payload into text, as the [orchestration page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_orchestration/) does for agent routing.

**Multi-selection.** `add_multi_selection_edge_group(source, targets, selection_func)` lets `selection_func(message, target_ids) -> list[str]` pick which targets receive each message:

```python
import asyncio
from dataclasses import dataclass
from typing_extensions import Never

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, executor, handler


@dataclass
class Task:
    description: str
    priority: str   # "high" | "low"


class Dispatcher(Executor):
    @handler
    async def dispatch(self, raw: str, ctx: WorkflowContext[Task]) -> None:
        priority = "high" if "urgent" in raw.lower() else "low"
        await ctx.send_message(Task(description=raw, priority=priority))


@executor(id="specialist-a")
async def specialist_a(task: Task, ctx: WorkflowContext[Never, str]) -> None:
    await ctx.yield_output(f"[A] {task.description}")


@executor(id="specialist-b")
async def specialist_b(task: Task, ctx: WorkflowContext[Never, str]) -> None:
    await ctx.yield_output(f"[B] {task.description}")


def route_by_priority(task: Task, target_ids: list[str]) -> list[str]:
    return target_ids if task.priority == "high" else target_ids[:1]


dispatcher = Dispatcher(id="dispatcher")
workflow = (
    WorkflowBuilder(start_executor=dispatcher)
    .add_multi_selection_edge_group(dispatcher, [specialist_a, specialist_b], route_by_priority)
    .build()
)


async def main() -> None:
    print((await workflow.run("Fix the login bug")).get_outputs())            # ['[A] Fix the login bug']
    print((await workflow.run("Urgent: production is down")).get_outputs())   # both specialists


asyncio.run(main())
```

**Conditional edges.** `add_edge(source, target, condition=fn)` sends the message only when `fn(message)` (sync or async) is truthy. A skipped edge is silent.

**Chains.** `add_chain([a, b, c])` is shorthand for `add_edge(a, b).add_edge(b, c)`.

### Choosing which executors produce output

By default every `ctx.yield_output(...)` call becomes an `output` event, which is why the fan-out example above returns the workers' progress notes as well as the aggregate. Pass `output_from=[...]` to the builder so only those executors' yields count as outputs, and `intermediate_output_from=[...]` (or `"all_other"`) to surface the others as `intermediate` events. Executors named in either list must declare a workflow output type (the second `WorkflowContext` parameter).

```python
workflow = (
    WorkflowBuilder(
        start_executor=splitter,
        output_from=[aggregator],
        intermediate_output_from=workers,
    )
    .add_fan_out_edges(splitter, workers)
    .add_fan_in_edges(workers, aggregator)
    .build()
)

result = await workflow.run("job")
print(result.get_outputs())               # ['job-A | job-B | job-C']
print(result.get_intermediate_outputs())  # ['a done', 'b done', 'c done']
```

The older `output_executors=` argument is deprecated (since 1.5.0) in favour of `output_from=`.

### Nesting workflows

`WorkflowExecutor(workflow, id=...)` turns a built workflow into a single node of another workflow. Create the executor once and use the same instance everywhere it appears in the outer graph; constructing two with the same id fails with `Duplicate executor ID`.

The node emits what the inner workflow outputs. Here that is the critic's `AgentResponse`, which an agent node does not accept, so a small executor passes the messages on:

```python
from agent_framework import Agent, AgentResponse, Message, WorkflowBuilder, WorkflowContext, WorkflowExecutor, executor
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
drafter = Agent(client=client, name="drafter", instructions="Draft a short article.")
critic = Agent(client=client, name="critic", instructions="Critique the draft.")
publisher = Agent(client=client, name="publisher", instructions="Produce the final text.")

inner = WorkflowBuilder(start_executor=drafter, name="draft-and-critique").add_edge(drafter, critic).build()



@executor(id="to-messages")
async def to_messages(response: AgentResponse, ctx: WorkflowContext[list[Message]]) -> None:
    await ctx.send_message(response.messages)


draft_pipeline = WorkflowExecutor(inner, id="draft-pipeline")
outer = (
    WorkflowBuilder(start_executor=draft_pipeline, name="publish-pipeline")
    .add_chain([draft_pipeline, to_messages, publisher])
    .build()
)

result = await outer.run("Write about heat pumps.")
print(result.get_outputs()[-1].text)
```

With `allow_direct_output=False` (the default), the inner workflow's outputs are sent on to the next node as messages; with `True` they become outputs of the outer workflow. A `request_info` inside the inner workflow reaches the parent as a `SubWorkflowRequestMessage` that a parent executor can answer; set `propagate_request=True` to pass it out to the caller instead.

### Visualising a workflow

```python
from agent_framework import WorkflowViz

viz = WorkflowViz(outer)
print(viz.to_mermaid())          # paste into Markdown
print(viz.to_digraph())          # Graphviz DOT
# viz.save_svg("workflow.svg")   # needs the graphviz package and the dot binary
```

Nested workflows render as clusters; `include_internal_executors=True` also shows the framework's internal nodes.

### Workflow events

`workflow.run(message, stream=True)` yields `WorkflowEvent` objects; `event.type` says what happened.

| `event.type` | Fields | Meaning |
|---|---|---|
| `started` | | The run began. |
| `status` | `state` | Lifecycle change: `IN_PROGRESS`, `IDLE`, `IDLE_WITH_PENDING_REQUESTS`, `FAILED`, `CANCELLED`. Included when `include_status_events=True`. |
| `output` / `intermediate` | `executor_id`, `data` | An executor called `ctx.yield_output(...)`. |
| `request_info` | `request_id`, `source_executor_id`, `data` | An executor is waiting for an external response. |
| `superstep_started` / `superstep_completed` | `iteration` | Superstep boundaries. |
| `executor_invoked` / `executor_completed` / `executor_failed` | `executor_id`, `details` | Per-executor lifecycle. |
| `executor_bypassed` | `executor_id` | A cached step result was replayed from a checkpoint. |
| `warning` / `error` / `failed` | `data`, `details` | Diagnostics; `failed` ends the run (`details` is a `WorkflowErrorDetails`). |

```python
async for event in workflow.run("job", stream=True):
    if event.type == "output":
        print(f"[{event.executor_id}] {event.data}")
    elif event.type == "executor_failed":
        print(f"{event.executor_id}: {event.details.error_type}: {event.details.message}")
```

`WorkflowRunResult` (a list of events) has `get_outputs()`, `get_intermediate_outputs()`, `get_request_info_events()` and `get_final_state()`.

### Checkpointing and human-in-the-loop

Give the builder a `CheckpointStorage` and every superstep is saved. Resume with `workflow.run(checkpoint_id=...)`:

```python
from agent_framework import FileCheckpointStorage, WorkflowBuilder

storage = FileCheckpointStorage("./checkpoints")
workflow = (
    WorkflowBuilder(start_executor=splitter, checkpoint_storage=storage, name="fan-pipeline")
    .add_fan_out_edges(splitter, workers)
    .add_fan_in_edges(workers, aggregator)
    .build()
)

latest = await storage.get_latest(workflow_name="fan-pipeline")
if latest:
    result = await workflow.run(checkpoint_id=latest.checkpoint_id)
```

Inside an executor, `await ctx.request_info(payload, response_type=bool)` pauses the workflow, and a method decorated with `@response_handler` on the same executor receives the answer when the caller runs `workflow.run(responses={request_id: value})`. Combined with checkpointing, a person can answer hours later in a different process. Storage backends (in-memory, file, Redis, Cosmos DB, custom) are on the [checkpointing page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_checkpointing/); the request/response loop, tool approval and plan review are on the [HITL page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_hitl/).

### A workflow as an agent

`workflow.as_agent(name=..., description=...)` returns a `WorkflowAgent` with the normal agent interface, so a whole graph can sit wherever an agent is expected: inside another workflow, behind `as_tool()`, or behind an HTTP endpoint.

```python
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
classifier = Agent(client=client, name="classifier", instructions="Tag the message.")
resolver = Agent(client=client, name="resolver", instructions="Answer the message.")

triage = WorkflowBuilder(start_executor=classifier, name="support-triage").add_edge(classifier, resolver).build()
triage_agent = triage.as_agent(name="support_triage", description="Classifies a ticket and answers it.")

supervisor = Agent(
    client=client,
    name="supervisor",
    instructions="Use the support_triage tool for support questions.",
    tools=[triage_agent.as_tool()],
)
response = await supervisor.run("My laptop won't charge.")
print(response.text)
```

Pending `request_info` calls inside the workflow surface on the agent's response as user-input requests, so the same approval-handling code works for single agents and workflows.

### An agent as an MCP server

`agent.as_mcp_server(server_name=..., version=..., instructions=..., lifespan=...)` returns an `mcp.server.lowlevel.Server` exposing the agent as one MCP tool (`<server_name>_run`). Connect it to any MCP transport:

```python
import asyncio
from mcp.server.stdio import stdio_server

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient


@tool
def search_inventory(sku: str) -> str:
    """Return the stock count for a SKU."""
    return f"SKU {sku}: 142 units in stock"


inventory_agent = Agent(
    client=OpenAIChatClient(),
    name="inventory-agent",
    instructions="Answer stock questions with search_inventory.",
    tools=[search_inventory],
)
mcp_server = inventory_agent.as_mcp_server(server_name="InventoryAgent", version="1.0.0")


async def run_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(read_stream, write_stream, mcp_server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run_stdio())
```

For HTTP, mount the server with the MCP SDK's streamable HTTP transport; [Recipe 16](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_recipes/#recipe-16--expose-an-agent-over-mcp) shows a complete server.

## Multi-agent orchestration

`agent-framework-orchestrations` provides five builders. Each one returns an ordinary `Workflow`, so streaming, checkpointing and HITL work the same way. Agents go in the keyword-only `participants=` argument.

| Pattern | Builder | Use |
|---|---|---|
| Sequential | `SequentialBuilder` | A pipeline: research, then analyse, then write. |
| Concurrent | `ConcurrentBuilder` | The same prompt to several agents in parallel, aggregated once. |
| Handoff | `HandoffBuilder` | A triage agent hands the conversation to specialists. |
| Group chat | `GroupChatBuilder` | Turn-taking panel, with speakers picked by a function or an orchestrator agent. |
| Magentic | `MagenticBuilder` | A manager agent plans, delegates, tracks progress and replans. |

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import ConcurrentBuilder, HandoffBuilder, SequentialBuilder

client = OpenAIChatClient()
researcher = Agent(client=client, name="researcher", instructions="Return concise facts on the topic.")
analyst = Agent(client=client, name="analyst", instructions="Turn the facts into an analysis.")
writer = Agent(client=client, name="writer", instructions="Write a one-paragraph summary.")

sequential = SequentialBuilder(participants=[researcher, analyst, writer]).build()


async def stitch(results) -> str:
    return "\n---\n".join(r.agent_response.text for r in results)

concurrent = ConcurrentBuilder(participants=[researcher, analyst, writer]).with_aggregator(stitch).build()

# Handoff participants must persist history after every model call.
triage = Agent(client=client, name="triage", instructions="Hand off to billing or technical.",
               require_per_service_call_history_persistence=True)
billing = Agent(client=client, name="billing", instructions="Resolve billing questions.",
                description="Invoices, refunds, plan changes.",
                require_per_service_call_history_persistence=True)
technical = Agent(client=client, name="technical", instructions="Resolve technical questions.",
                  description="Bugs, API errors, outages.",
                  require_per_service_call_history_persistence=True)
handoff = (
    HandoffBuilder(participants=[triage, billing, technical])
    .with_start_agent(triage)
    .add_handoff(triage, [billing, technical])
    .build()
)


async def main() -> None:
    result = await sequential.run("Post-quantum cryptography")
    print(result.get_outputs()[-1])


asyncio.run(main())
```

Group chat, Magentic, plan review, autonomous handoff mode and the request-info hooks are covered on the [orchestration page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_orchestration/).

## MCP

MCP servers are tool sources. Use `MCPStdioTool` for local processes, `MCPStreamableHTTPTool` for HTTP servers and `MCPWebsocketTool` for websocket servers, each as an async context manager:

```python
import asyncio
from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    async with MCPStreamableHTTPTool(
        name="learn",
        url="https://learn.microsoft.com/api/mcp",
        description="Search Microsoft Learn documentation.",
    ) as learn:
        agent = Agent(client=OpenAIChatClient(), instructions="Answer with the learn tool.", tools=learn)
        print((await agent.run("How does DefaultAzureCredential pick a credential?")).text)


asyncio.run(main())
```

`header_provider=lambda kwargs: {...}` builds per-request headers from the `function_invocation_kwargs` of the `run()` call, which suits multi-tenant tokens. `allowed_tools`, `approval_mode`, `tool_name_prefix`, sampling and long-running MCP tasks are on the [MCP page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_mcp/).

## Agent harness

The harness providers give an agent planning and working-memory tools. Add them as context providers, or use `create_harness_agent(...)` to get an agent with all of them wired together.

| Provider | Gives the agent | Status in %%version=agent-framework%% |
|---|---|---|
| `TodoProvider` (with `TodoSessionStore`) | A task list: `todos_add`, `todos_complete`, `todos_remove`, `todos_get_remaining`, `todos_get_all` | Stable since 1.12.0 |
| `TodoFileStore` | Todos persisted to disk | Experimental (`HARNESS`) |
| `AgentModeProvider` | Named modes (default `plan` / `execute`) with `mode_get` / `mode_set` tools | Stable since 1.12.0 |
| `MemoryContextProvider` + `MemoryFileStore` | Long-term memory extracted from conversations and injected into later sessions | Experimental (`HARNESS`) |
| `FileMemoryProvider` | Session-scoped scratch files | Stable |
| `FileAccessProvider` | A shared workspace with read, write, grep and replace tools | Experimental (`HARNESS`) |
| `BackgroundAgentsProvider` | Delegation to sub-agents running concurrently | Experimental (`HARNESS`) |
| `ToolApprovalMiddleware` | Standing "always approve" rules | Stable |
| `AgentLoopMiddleware` | Re-runs the agent until a condition or judge says the task is done | Experimental (`HARNESS`) |

Experimental APIs emit an `ExperimentalWarning` on first use and may change between minor releases. Examples for every provider are on the [harness page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_harness/).

## Vector stores

> **Experimental** (`ExperimentalFeature.VECTOR_STORES`). The model, collection and filter layer arrived in 1.18.0; `VectorStoreHistoryProvider`, `VectorCollectionContextProvider` and the `create_*_tool` factories in 1.19.0.

`agent_framework` has a provider-neutral vector-store layer. Describe a record type once, get a collection for it from a store (the dependency-free `InMemoryStore` here; `agent-framework-azure-ai-search`, `agent-framework-redis` and `agent-framework-azure-cosmos` provide stores with the same interface), then query it from code or hand it to an agent as tools.

### Describing records

Annotate a dataclass or Pydantic model. A model needs exactly one `"key"` field; other fields are `"data"` unless marked `"vector"`. Vector fields need `dimensions` and a default (`= None`), because reads leave vectors out unless you pass `include_vectors=True`.

```python
from dataclasses import dataclass
from typing import Annotated
from agent_framework import VectorStoreField, vectorstoremodel


@vectorstoremodel(collection_name="products")
@dataclass
class Product:
    sku: Annotated[str, VectorStoreField("key")]
    name: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    category: Annotated[str, VectorStoreField("data", is_indexed=True)]
    price: Annotated[float, VectorStoreField("data")]
    embedding: Annotated[list[float] | None, VectorStoreField(
        "vector", dimensions=3, distance_function="cosine_similarity", index_kind="hnsw",
    )] = None


defn = Product.__vectorstoremodel_definition__
print(defn.key_name, defn.vector_field_names, defn.data_field_names)
# sku ['embedding'] ['name', 'category', 'price']
```

Other `VectorStoreField` options are `storage_name`, `is_auto_generated`, a per-field `embedding_generator` and `provider_annotations`. For `dict` records, build a `VectorStoreCollectionDefinition(fields=[...], collection_name=...)` and pass it as `definition=`.

### Reading, writing and searching

```python
import asyncio
from agent_framework import Filter, FilterGroup, InMemoryStore


async def main() -> None:
    store = InMemoryStore()                       # embedding_generator=... to embed text for you
    products = store.get_collection(Product)
    await products.ensure_collection_exists()

    await products.upsert([
        Product("A1", "Noise-cancelling headphones", "electronics", 299.0, [1.0, 0.0, 0.0]),
        Product("B2", "USB-C cable", "electronics", 9.99, [0.9, 0.1, 0.0]),
        Product("C3", "Field guide to birds", "books", 24.5, [0.0, 0.0, 1.0]),
    ], generate_vectors=False)

    print([p.name for p in await products.get(["A1"])])

    cheap_electronics = FilterGroup("and", [Filter("category", "eq", "electronics"), Filter("price", "lt", 50.0)])
    print([p.sku for p in await products.get(filter=cheap_electronics, top=10)])   # ['B2']

    results = await products.search(vector=[1.0, 0.0, 0.0], top=2)
    async for hit in results.results:
        print(hit["record"].sku, round(hit["score"], 3))


asyncio.run(main())
```

`get(keys=None, *, filter=None, top=10, skip=0, order_by=None, include_vectors=False)` reads by key or lists a filtered page. `search(values=None, *, vector=None, filter=None, top=3, score_threshold=None, ...)` takes a vector, or text that the collection's embedding generator embeds. `InMemoryCollection` is neither persistent nor thread-safe; use it for tests and prototypes.

`Filter(field, operator, value)` supports `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `between`, `in`, `not_in`, `is_null`, `is_not_null`, `exists`, `contains`, `contains_any`, `contains_all`, `starts_with`, `ends_with` and `contains_text`. `FilterGroup("and" | "or" | "not", [...])` combines them, and each connector translates the expression into its own query language.

### Vector search as agent tools

`Param(...)` in place of a filter value lets the model fill it in when the filter belongs to a tool:

```python
from agent_framework import Filter, InMemoryStore, Param, create_vector_search_tool

products = InMemoryStore().get_collection(Product)
search_tool = create_vector_search_tool(
    products,
    name="search_products",
    description="Semantic product search; optionally cap the price.",
    filter=Filter("price", "lte", Param("max_price", float | None, default=None, omit_if_none=True)),
    top=5,
)
```

`create_get_tool`, `create_upsert_tool` and `create_delete_tool` cover the other operations; upsert and delete default to `approval_mode="always_require"`. `VectorCollectionContextProvider(collection, scope_filter=...)` generates up to four of these tools for an agent at once. `scope_filter` is ANDed into every tool's query; it separates data but is not an authorisation boundary. To keep the agent's own conversation in a vector store, use [`VectorStoreHistoryProvider`](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sessions/#vectorstorehistoryprovider--history-in-a-vector-store).

## Prompt-injection defence (FIDES)

> **Experimental** (`ExperimentalFeature.FIDES`) in %%version=agent-framework%%. Import from `agent_framework.security`.

FIDES applies information-flow control to tool calls. Every tool result carries an integrity label (`TRUSTED` / `UNTRUSTED`) and a confidentiality label (`PUBLIC` / `PRIVATE` / `USER_IDENTITY`). Labels propagate through the run, and once untrusted content is in context, privileged tools are blocked or sent for human approval. Untrusted results can also be hidden from the model behind variable references, which the agent inspects only through a separate quarantined model call.

`SecureAgentConfig` is a context provider that installs the whole stack: `LabelTrackingFunctionMiddleware`, `PolicyEnforcementFunctionMiddleware`, and the `quarantined_llm` and `inspect_variable` tools.

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from agent_framework.security import ConfidentialityLabel, IntegrityLabel, SecureAgentConfig


@tool
async def fetch_news(query: str) -> str:
    """Fetch news headlines. External content, so untrusted."""
    return f"[external] News about {query}"


@tool
async def send_email(to: str, body: str) -> str:
    """Send an email. Must not be triggered by untrusted content."""
    return f"sent to {to}"


@tool
async def log_search(query: str) -> str:
    """Record the search for auditing. Safe in untrusted context."""
    return f"logged: {query}"


security = SecureAgentConfig(
    default_integrity=IntegrityLabel.UNTRUSTED,       # tool results are untrusted unless a tool says otherwise
    default_confidentiality=ConfidentialityLabel.PUBLIC,
    allow_untrusted_tools={"log_search"},             # may still run once context is untrusted
    block_on_violation=False,
    approval_on_violation=True,                       # ask a person instead of blocking outright
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a research assistant.",
    tools=[fetch_news, send_email, log_search],
    context_providers=[security],
)


async def main() -> None:
    response = await agent.run("Find news about AI and email a summary to boss@example.com.")
    print(response.text)
    for request in response.user_input_requests:     # approval requests raised by the policy
        print("needs approval:", request)


asyncio.run(main())
```

| `SecureAgentConfig` parameter | Default | Effect |
|---|---|---|
| `auto_hide_untrusted` | `True` | Replace untrusted results with variable references the model can't read directly. |
| `default_integrity` / `default_confidentiality` | `UNTRUSTED` / `PUBLIC` | Labels for tool results that don't declare their own. |
| `allow_untrusted_tools` | `None` | Tool names allowed to run in an untrusted context. |
| `block_on_violation` / `approval_on_violation` | `True` / `False` | Block a violating call, or turn it into an approval request. |
| `enable_audit_log` / `enable_policy_enforcement` | `True` / `True` | Audit logging; the policy middleware itself. |
| `quarantine_chat_client` | `None` | Separate model used by `quarantined_llm`. |
| `max_pending_approvals` / `pending_approval_ttl` | `256` / 1 hour | Limits on outstanding approval requests. |

A tool declares its own trust with `@tool(additional_properties={"source_integrity": "trusted"})`. For MCP servers, `SecureMCPToolProxy` labels the server's tools locally so the same policy applies; provider-hosted MCP runs on the provider's side and bypasses local middleware entirely. Content Safety and Purview, which complement FIDES, are on the [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/).

## A2A

`agent-framework-a2a` connects to remote Agent2Agent agents (`A2AAgent`) and serves framework agents over A2A (`A2AExecutor`). Both directions are on the [A2A page](/microsoft-agent-framework-guide/microsoft_agent_framework_a2a_protocol/).

## Hosting and deployment

| Target | Package | Notes |
|---|---|---|
| Local debugging UI | `agent-framework-devui` | `devui` CLI or `serve(entities=[...])`; development only. |
| Microsoft Foundry hosted agents | `agent-framework-foundry-hosting` | `ResponsesHostServer` serves an agent with the OpenAI Responses contract. |
| Azure Functions / Durable Task | `agent-framework-azurefunctions`, `agent-framework-durabletask` | Durable agents whose history survives restarts. |
| Your own web app | `agent-framework-core` | Reuse one agent per process; see the production page for FastAPI patterns. |

All four are described on the [production page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_production_guide/), with observability on the [observability page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_observability/) and evaluation in CI on the [evaluation page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_evaluation/).

## API map

Where each part of the public API is documented. Classes are importable from `agent_framework` unless a module is given.

| Area | Main classes and functions | Page |
|---|---|---|
| Agents | `Agent`, `RawAgent`, `BaseAgent`, `AgentResponse`, `AgentResponseUpdate`, `ResponseStream` | This page; [Advanced patterns](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_advanced/) |
| Chat clients | `OpenAIChatClient`, `OpenAIChatCompletionClient`, `FoundryChatClient`, `AnthropicClient`, `BedrockChatClient`, `OllamaChatClient`, `GeminiChatClient`, `ClaudeAgent`, `GitHubCopilotAgent`, `CopilotStudioAgent` | [Model providers](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_model_providers/) |
| Custom clients and providers | `BaseChatClient`, `BaseEmbeddingClient`, `ContextProvider`, `Supports*` protocols, `Content` factories | [Advanced patterns](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_advanced/) |
| Tools | `@tool`, `FunctionTool`, `FunctionInvocationConfiguration` | [Tools](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_tools/) |
| MCP | `MCPStdioTool`, `MCPStreamableHTTPTool`, `MCPWebsocketTool`, `MCPTaskOptions`, `Agent.as_mcp_server()` | [MCP](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_mcp/) |
| Skills | `InlineSkill`, `ClassSkill`, `FileSkillsSource`, `SkillsProvider`, `*SkillsSource` | [Skills](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_skills/) |
| Sessions | `AgentSession`, `HistoryProvider`, `InMemoryHistoryProvider`, `FileHistoryProvider`, `VectorStoreHistoryProvider`, `FileSessionStore` | [Sessions](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sessions/) |
| Compaction | `CompactionProvider` and the strategy classes | [Compaction](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_compaction/) |
| Middleware | `AgentMiddleware`, `ChatMiddleware`, `FunctionMiddleware`, contexts, `MiddlewareBundle`, `MessageInjectionMiddleware` | [Middleware](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_middleware/) |
| Harness | `TodoProvider`, `AgentModeProvider`, `MemoryContextProvider`, `FileMemoryProvider`, `FileAccessProvider`, `BackgroundAgentsProvider`, `ToolApprovalMiddleware`, `AgentLoopMiddleware`, `create_harness_agent` | [Harness](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_harness/) |
| Graph workflows | `WorkflowBuilder`, `Executor`, `@executor`, `@handler`, `AgentExecutor`, `WorkflowExecutor`, `Case` / `Default`, `WorkflowViz`, `WorkflowEvent` | This page |
| Functional workflows | `@workflow`, `@step`, `RunContext` | [Functional workflows](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_functional_workflows/) |
| Checkpointing | `CheckpointStorage`, `FileCheckpointStorage`, `InMemoryCheckpointStorage`, `WorkflowCheckpoint` | [Checkpointing](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_checkpointing/) |
| Human in the loop | `ctx.request_info`, `@response_handler`, tool approval, `AgentRequestInfoResponse` | [HITL](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_hitl/) |
| Orchestration (`agent_framework_orchestrations`) | `SequentialBuilder`, `ConcurrentBuilder`, `HandoffBuilder`, `GroupChatBuilder`, `MagenticBuilder` | [Orchestration](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_orchestration/) |
| Vector stores | `@vectorstoremodel`, `VectorStoreField`, `InMemoryStore`, `Filter`, `FilterGroup`, `Param`, `VectorCollectionContextProvider` | This page |
| Security (`agent_framework.security`) | `SecureAgentConfig`, `LabelTrackingFunctionMiddleware`, `PolicyEnforcementFunctionMiddleware`, `SecureMCPToolProxy`, `IntegrityLabel`, `ConfidentialityLabel` | This page |
| Evaluation | `LocalEvaluator`, `evaluate_agent`, `evaluate_workflow`, `EvalItem`, `EvalResults`, `@evaluator`, `keyword_check`, `tool_called_check` | [Evaluation](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_evaluation/) |
| Observability (`agent_framework.observability`) | `configure_otel_providers`, `enable_instrumentation`, `ExperimentalFeature` | [Observability](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_observability/) |
| Settings | `load_settings`, `SecretString` | This page |
| Declarative (`agent_framework.declarative`) | `AgentFactory`, `WorkflowFactory` | [Workflows & declarative agents](/microsoft-agent-framework-guide/microsoft_agent_framework_graphs_declarative/) |
| A2A (`agent_framework.a2a`) | `A2AAgent`, `A2AAgentSession`, `A2AExecutor` | [A2A](/microsoft-agent-framework-guide/microsoft_agent_framework_a2a_protocol/) |
| Errors (`agent_framework.exceptions`) | `AgentFrameworkException` and subclasses | [Advanced patterns](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_advanced/#exception-hierarchy) |

## Revision history

| Date | Version | Changes |
|---|---|---|
| 2026-10-21 | 1.22.0 | Rewritten. Removed the separate class reference (its entries now live on the topic pages, indexed by the API map) and the duplicated orchestration, harness and session sections. Fixed streaming (`get_final_response()`), structured output (`options={"response_format": ...}`), `WorkflowContext[Never, ...]`, nested-workflow executor reuse and the per-call `tools=` description. Added the A2A, hosting and API map sections. |
| 2026-08-21 | 1.14.0 | Per-class reference pages consolidated into this guide. |
| 2026-05-15 | 1.4.0 | `SecureAgentConfig` (FIDES) section added. |
| 2026-05 | 1.2.2 | Skills, functional workflows and `Agent.as_tool()` added. |

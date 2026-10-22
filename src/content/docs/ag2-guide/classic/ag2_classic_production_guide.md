---
title: "AG2 Classic Production Guide"
description: "Running AG2 Classic (import autogen) in production: pinning, offline tests with TestAgent, sandboxed code execution, caching, cost tracking, runtime logging, OpenTelemetry, concurrency and serving over A2A."
framework: "ag2"
language: "python"
sidebar:
  label: "Production"
---

> **This page covers AG2 Classic**, the `import autogen` API published on PyPI as `autogen` (and as `ag2` up to 0.14), checked against `autogen` %%version=autogen%%. Classic is in maintenance mode (security and critical fixes only); for new work use [AG2 1.x](/ag2-guide/ag2_production_guide/). Every Python example here was run against the installed package with `autogen.testing.TestAgent` or a local server in place of a real model.

## Pin the distribution

The trap with Classic is the package name. `ag2>=1.0` no longer contains the `autogen` module, and `pyautogen` now installs Microsoft's `autogen-agentchat`. Pin `autogen` with an upper bound:

```text
# requirements.txt
autogen[openai]>=%%version=autogen%%,<0.15
# extras used on this page: docker, tracing, a2a, interop-langchain
```

`autogen` and `ag2<1.0` both install the same `autogen` module and overwrite each other, so keep only one of them in an environment. See the [migration guide](/ag2-guide/classic/ag2_classic_migration_guide/) for the other package-name pitfalls.

## Test without a model

`autogen.testing.TestAgent` swaps an agent's LLM client for a scripted one for the duration of a `with` block. Script plain strings for answers and `tools_message(ToolCall(...))` for tool calls, so tests exercise your tools, termination logic and message flow offline:

```python
from autogen import ConversableAgent, LLMConfig, register_function
from autogen.testing import TestAgent, ToolCall, tools_message

llm_config = LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"})


def lookup_order(order_id: str) -> str:
    """Look up an order's status."""
    return f"order {order_id}: shipped"


assistant = ConversableAgent("assistant", system_message="Answer order questions with your tools.", llm_config=llm_config)
user = ConversableAgent("user", human_input_mode="NEVER", llm_config=False)
register_function(lookup_order, caller=assistant, executor=user, description="Look up an order's status.")


def test_order_lookup() -> None:
    script = [tools_message(ToolCall("lookup_order", order_id="42")), "Order 42 has shipped."]
    with TestAgent(assistant, script):
        result = user.initiate_chat(assistant, message="Where is order 42?", max_turns=2, silent=True)

    contents = [m.get("content") for m in result.chat_history]
    assert "order 42: shipped" in contents          # the real tool ran
    assert contents[-1] == "Order 42 has shipped."


test_order_lookup()
```

`TestAgent` also sets `human_input_mode="NEVER"` while active. `LLMConfig` still builds a real client when the agent is created, so set `OPENAI_API_KEY` to any value in CI. Keep a small suite that calls the real provider and run it on a schedule.

## Code execution

- Run model-written code in Docker: `code_execution_config={"executor": DockerCommandLineCodeExecutor(image="python:3.12-slim", work_dir="workspace", timeout=60)}` from `autogen.coding` (extra `autogen[docker]`, plus a running Docker daemon). The [comprehensive guide](/ag2-guide/classic/ag2_classic_comprehensive_guide/#code-execution) has a full example.
- Use `LocalCommandLineCodeExecutor` only for trusted code. It runs with the permissions of your process.
- `UserProxyAgent` turns code execution on by default and tries Docker. Agents that only relay messages or run registered tools should pass `code_execution_config=False`. Without it they raise `RuntimeError` on a machine with no Docker daemon.
- Executors take a `timeout`. Bound conversations with `max_turns` on `initiate_chat` and `max_round` on `GroupChat`.

## Caching

Classic can cache LLM responses by seed. Caching is off by default (`cache_seed=None`). Set `cache_seed=42` in `LLMConfig` to use an on-disk cache under `.cache/`, or pass an explicit `Cache` to the chat:

```python
from autogen import Cache, ConversableAgent, LLMConfig
from autogen.testing import TestAgent

llm_config = LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"})
assistant = ConversableAgent("assistant", llm_config=llm_config)
user = ConversableAgent("user", human_input_mode="NEVER", llm_config=False)

with TestAgent(assistant, ["Paris."]):                      # remove outside tests
    with Cache.disk(cache_seed=42, cache_path_root=".cache") as cache:
        result = user.initiate_chat(assistant, message="Capital of France?", max_turns=1, cache=cache, silent=True)
print(result.summary)
```

`Cache.redis(cache_seed=..., redis_url=...)` shares the cache between processes (extra `autogen[redis]`). Turn caching off for anything where a fresh answer matters. Identical prompts return the cached reply even when the world has changed.

## Cost and usage

Every `ChatResult` carries `cost`, split into usage including and excluding cached responses. `gather_usage_summary` totals it across agents:

```python
from autogen import gather_usage_summary

# result = user.initiate_chat(assistant, ...)
print(result.cost["usage_excluding_cached_inference"])
print(gather_usage_summary([assistant, user])["usage_excluding_cached_inference"]["total_cost"])
```

Costs come from the client's price table. For models it does not know, set `price=[prompt_per_1k, completion_per_1k]` on the config entry.

## Runtime logging

`autogen.runtime_logging` records agents, LLM calls, tool calls and events to SQLite (or a JSON-lines file with `logger_type="file"`), keyed by a session ID:

```python
import sqlite3

import autogen

session_id = autogen.runtime_logging.start(config={"dbname": "logs.db"})
# ... run chats ...
autogen.runtime_logging.stop()

con = sqlite3.connect("logs.db")
print([row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")])
# ['chat_completions', 'agents', 'oai_wrappers', 'oai_clients', 'version', 'events', 'function_calls']
```

`chat_completions` stores request, response, cost and latency per LLM call. It is the quickest way to answer "what did the model see?" after an incident.

## Tracing with OpenTelemetry

`autogen.opentelemetry` (extra `autogen[tracing]`) emits spans that follow the OpenTelemetry GenAI conventions. Instrument the LLM wrapper once, then each agent (and `instrument_pattern(pattern, ...)` for group-chat patterns):

```python
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from autogen import ConversableAgent, LLMConfig
from autogen.opentelemetry import instrument_agent, instrument_llm_wrapper

tracer_provider = TracerProvider(resource=Resource.create({"service.name": "support-bot"}))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))  # use an OTLP exporter in production
instrument_llm_wrapper(tracer_provider=tracer_provider)

llm_config = LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"})
assistant = instrument_agent(ConversableAgent("assistant", llm_config=llm_config), tracer_provider=tracer_provider)
user = instrument_agent(ConversableAgent("user", human_input_mode="NEVER", llm_config=False), tracer_provider=tracer_provider)
```

A chat then produces `conversation user` and `invoke_agent assistant` spans, with LLM calls nested beneath. `instrument_llm_wrapper(..., capture_messages=True)` adds prompts and completions to the spans; leave it off where messages contain personal data. `instrument_a2a_server` covers A2A servers.

## Resilience

- Give `LLMConfig` several entries. The client tries them in order and moves to the next on errors such as rate limits (`routing_method="round_robin"` spreads load instead).
- Set `timeout` and `max_retries` on OpenAI-family entries: `LLMConfig({"api_type": "openai", "model": "...", "timeout": 30, "max_retries": 2})`.
- Tool exceptions are caught and returned to the model as the tool result, so the conversation continues. Raise only for failures the model cannot recover from, and validate tool arguments yourself.
- Use `is_termination_msg=lambda m: (m.get("content") or "").rstrip().endswith("TERMINATE")`. Tool-call messages have `content=None`.

## Concurrency

Agents keep conversation state, so give each concurrent conversation its own agent instances and use the async API:

```python
import asyncio

from autogen import ConversableAgent, LLMConfig
from autogen.testing import TestAgent

llm_config = LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"})


async def answer(question: str) -> str:
    assistant = ConversableAgent("assistant", llm_config=llm_config)
    user = ConversableAgent("user", human_input_mode="NEVER", llm_config=False)
    with TestAgent(assistant, [f"Answer to: {question}"]):   # remove outside tests
        result = await user.a_initiate_chat(assistant, message=question, max_turns=1, silent=True)
    return result.summary


async def main() -> None:
    print(await asyncio.gather(*(answer(q) for q in ["What is DNS?", "What is TLS?", "What is HTTP/3?"])))


asyncio.run(main())
```

Bound concurrency with an `asyncio.Semaphore` to stay inside provider rate limits.

## Serving an agent over A2A

`autogen.a2a` (extra `autogen[a2a]`, marked experimental upstream) wraps a `ConversableAgent` as an A2A server and a remote A2A agent as a chat participant. The server is a Starlette app, so run it under any ASGI server:

```python
# server.py — run with: uvicorn server:app --host 0.0.0.0 --port 8000
from a2a.compat.v0_3.types import AgentSkill

from autogen import ConversableAgent, LLMConfig
from autogen.a2a import A2aAgentServer, CardSettings

reviewer = ConversableAgent(
    "reviewer",
    system_message="You review Python code and reply with concrete fixes.",
    llm_config=LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"}),
)

card = CardSettings(
    description="Reviews Python snippets.",
    skills=[AgentSkill(id="review", name="Code review", description="Reviews Python code.", tags=["python"])],
)
app = A2aAgentServer(reviewer, url="http://localhost:8000", agent_card=card).build()
```

The card is served at `/.well-known/agent-card.json`. On the client side, `A2aRemoteAgent` only supports the async API:

```python
import asyncio

from autogen import ConversableAgent
from autogen.a2a import A2aRemoteAgent


async def main() -> None:
    remote = A2aRemoteAgent(url="http://localhost:8000", name="reviewer")
    user = ConversableAgent("user", human_input_mode="NEVER", llm_config=False)
    result = await user.a_initiate_chat(remote, message="Review: f = open('x'); f.read()", max_turns=1)
    print(result.summary)


asyncio.run(main())
```

Calling the synchronous `initiate_chat` with a remote agent raises `NotImplementedError`. `A2aAgentServer.add_middleware(...)` adds Starlette middleware (authentication, CORS) to the app.

## Using LangChain tools

`autogen.interop.Interoperability` converts tools from other frameworks into Classic `Tool`s (extra `autogen[interop-langchain]`):

```python
from langchain_core.tools import tool as langchain_tool

from autogen import ConversableAgent, LLMConfig
from autogen.interop import Interoperability


@langchain_tool
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


assistant = ConversableAgent("assistant", llm_config=LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"}))
user = ConversableAgent("user", human_input_mode="NEVER", llm_config=False)

tool = Interoperability.convert_tool(tool=word_count, type="langchain")
tool.register_for_llm(assistant)
tool.register_for_execution(user)
```

The same works for prebuilt `langchain_community` tools. For retrieval, Classic has its own `RetrieveUserProxyAgent` (extra `autogen[retrievechat]`). You can also call any retriever from inside an ordinary registered function.

## Deployment

Classic has no deployment layer of its own. Chats are ordinary Python calls, so wrap them in your web framework or task queue, create fresh agents per request (see [Concurrency](#concurrency)), and keep provider keys in the environment or a secrets manager. The A2A server above is the only built-in way to expose an agent over HTTP.

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | autogen 0.14.1 | Moved from `/autogen-guide/python/autogen_production_guide/` and rewritten around Classic's own production features (TestAgent, caching, usage summaries, runtime logging, `autogen.opentelemetry`, async chats, A2A, Interoperability). Removed the broken A2A server/client code, the LangChain/LlamaIndex section built on removed APIs, the `ag2[openai]==0.2.x` pin and helper classes that were never wired in. |

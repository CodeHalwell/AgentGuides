---
title: "Haystack Advanced Agent Patterns (Python)"
description: "Hooks, state handlers, structured output, custom async components and resilient tool design for Haystack 3 agents."
framework: "haystack"
language: "python"
---

> Verified against `haystack-ai` %%version=haystack-ai%% (checked %%versions-checked%%). Upstream: [release notes](https://github.com/deepset-ai/haystack/releases)

This page assumes you have read [Creating Agents](/haystack-guide/haystack_comprehensive_guide/#creating-agents-with-the-agent-class) and [Agent Configuration](/haystack-guide/haystack_comprehensive_guide/#agent-configuration-and-customisation).

## Token budgets

`max_agent_steps` bounds the number of loop iterations, not what they cost. To cap spending, use the built-in `TokenBudgetHook` (experimental; it emits an `ExperimentalWarning`). It runs at `before_llm`, reads the cumulative `token_usage` the Agent records, and ends the run with `exit_reason="token_budget_exceeded"` before the next model call, keeping the messages collected so far:

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.hooks.budget import TokenBudgetHook
from haystack.tools import tool


@tool
def slow_lookup(key: str) -> str:
    """Look up a value."""
    return f"value for {key}"


agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[slow_lookup],
    hooks={"before_llm": [TokenBudgetHook(max_total_tokens=20_000, add_final_message=True)]},
)
```

Only the Agent's own chat-generator calls count; tokens spent inside tools (including `AgentTool` sub-agents) or by other hooks do not. Give each sub-agent its own budget. To keep long runs within the context window rather than a spending limit, see [context management](/haystack-guide/haystack_comprehensive_guide/#context-management).

## Stateful hooks

The `@hook` decorator suits simple functions. For hooks that hold configuration or resources, implement the `Hook` protocol directly: a `run(self, state)` method, and optionally `run_async(self, state)`. `TokenBudgetHook` is written this way; here are two more:

```python
import logging
import time

from haystack.components.agents import State
from haystack.hooks import FunctionHook

logger = logging.getLogger(__name__)


class ToolCallLimit:
    """Stop the run once the agent has made too many tool calls in total."""

    def __init__(self, max_calls: int):
        self.max_calls = max_calls

    def run(self, state: State) -> None:
        calls = sum((state.get("tool_call_counts") or {}).values())
        if calls >= self.max_calls:
            logger.warning("Tool-call limit reached (%d)", calls)
            state.set("stop_run", "tool_call_limit")      # becomes the run's exit_reason


class StepTimer:
    """Time each batch of tool calls, using one hook object per hook point."""

    def __init__(self):
        self._started = 0.0
        self.start = FunctionHook(function=self._start)
        self.stop = FunctionHook(function=self._stop)

    def _start(self, state: State) -> None:
        self._started = time.perf_counter()

    def _stop(self, state: State) -> None:
        logger.info("step %s tools took %.2fs", state.get("step_count"), time.perf_counter() - self._started)


timer = StepTimer()

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[slow_lookup],
    hooks={
        "before_llm": [TokenBudgetHook(max_total_tokens=20_000), ToolCallLimit(max_calls=10)],
        "before_tool": [timer.start],
        "after_tool": [timer.stop],
    },
)
```

Hook points, in order of a run: `before_run` → (`before_llm` → model call → `before_tool` → tools → `after_tool`)* → `on_exit` → `after_run`. In `on_exit`, setting `continue_run` to `True` sends the agent round the loop again; setting `stop_run` in any hook that runs before the next LLM call ends the run with that string as `exit_reason`.

## State handlers

Each `state_schema` entry can name a `handler(current, new)` that decides how a tool's output is merged. Lists are appended by default; everything else is replaced.

```python
def merge_scores(current: dict | None, new: dict) -> dict:
    merged = dict(current or {})
    for k, v in new.items():
        merged[k] = max(v, merged.get(k, v))
    return merged


@tool(outputs_to_state={"scores": {"source": "scores"}})
def score_source(url: str, score: float) -> dict:
    """Record a credibility score for a source URL."""
    return {"scores": {url: score}}


agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[score_source],
    state_schema={"scores": {"type": dict, "handler": merge_scores}},
)
```

Initial values for state keys are passed as keyword arguments to `run()`: `agent.run(messages=[...], scores={"https://example.com": 0.9})`.

## Structured output

Ask the chat generator for JSON that matches a Pydantic model with `response_format`, then validate the final message:

```python
from pydantic import BaseModel


class Verdict(BaseModel):
    answer: str
    confidence: float
    sources: list[str]


agent = Agent(
    chat_generator=OpenAIChatGenerator(
        model="%%model=openai-small%%",
        generation_kwargs={"response_format": Verdict},
    ),
    system_prompt="Answer the question and give your confidence from 0 to 1.",
)

result = agent.run(messages=[ChatMessage.from_user("Is the Great Wall visible from orbit?")])
verdict = Verdict.model_validate_json(result["last_message"].text)
```

Alternatively give the agent a `final_answer` tool whose parameters are the fields you need and set `exit_conditions=["final_answer"]`; this works with any tool-calling model.

## Custom components with async and lifecycle

Components that hold clients should create them in `warm_up()` and release them in `close()` (sync) and `close_async()` (async); `pipeline.close()` and `await pipeline.close_async()` call them. Provide `run_async` for non-blocking I/O.

```python
import httpx

from haystack import component


@component
class ExchangeRates:
    def __init__(self, base_url: str = "https://api.example.com"):
        self.base_url = base_url
        self._client: httpx.Client | None = None
        self._aclient: httpx.AsyncClient | None = None

    def warm_up(self) -> None:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=10)
            self._aclient = httpx.AsyncClient(base_url=self.base_url, timeout=10)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        self.close()
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None

    @component.output_types(rate=float)
    def run(self, base: str, quote: str):
        self.warm_up()
        resp = self._client.get("/rates", params={"base": base, "quote": quote})
        resp.raise_for_status()
        return {"rate": resp.json()["rate"]}

    @component.output_types(rate=float)
    async def run_async(self, base: str, quote: str):
        self.warm_up()
        resp = await self._aclient.get("/rates", params={"base": base, "quote": quote})
        resp.raise_for_status()
        return {"rate": resp.json()["rate"]}
```

Wrap it with `ComponentTool(component=ExchangeRates(), name="fx_rate", description="...")` to make it callable by an agent; the agent uses `run_async` when it runs asynchronously.

## Resilient tool design

- **Return, don't raise, for expected failures.** "No order with that ID" is information the model can use; a stack trace is not.
- **Keep schemas small.** Every parameter is a chance for the model to get it wrong. Prefer `Literal[...]` over free text for enumerations.
- **Keep privileged inputs away from the model.** `inputs_from_state` is a convenience, not a security boundary: arguments the model supplies override the state lookup. Read tenant or user IDs from a `State`-typed tool parameter (`def search(query: str, state: State)` then `state.get("tenant_id")`), which the Agent always injects itself, or from `hook_context` in hooks. See the [multi-tenant recipe](/haystack-guide/haystack_recipes/#multi-tenant-customer-support-agent).
- **Make side effects idempotent** (accept a client-generated request ID) so a retried tool call is harmless, and put a `ConfirmationHook` in front of anything irreversible.
- **Bound concurrency.** The agent may call several tools in one step; `tool_concurrency_limit` (default 4) caps how many run at once.

## Hybrid retrieval with a fallback

A pipeline that joins BM25 and embedding results, and an agent that falls back to web search when the knowledge base has nothing relevant:

```python
from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import PipelineTool

store = InMemoryDocumentStore()   # assume documents were indexed with OpenAIDocumentEmbedder

hybrid = Pipeline()
hybrid.add_component("embed", OpenAITextEmbedder())
hybrid.add_component("bm25", InMemoryBM25Retriever(document_store=store, top_k=10))
hybrid.add_component("dense", InMemoryEmbeddingRetriever(document_store=store, top_k=10))
hybrid.add_component("join", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=5))
hybrid.connect("embed.embedding", "dense.query_embedding")
hybrid.connect("bm25.documents", "join.documents")
hybrid.connect("dense.documents", "join.documents")

kb_search = PipelineTool(
    pipeline=hybrid,
    name="kb_search",
    description="Search the internal knowledge base. Returns an empty result if nothing matches.",
    input_mapping={"query": ["embed.text", "bm25.query"]},
    output_mapping={"join.documents": "documents"},
)


@tool
def web_search(query: str) -> str:
    """Search the public web. Use only if kb_search found nothing relevant."""
    return f"(stub) web results for {query!r}"


agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[kb_search, web_search],
    system_prompt="Prefer the knowledge base. Say which source each fact came from.",
)
```

For tracing and dead-letter handling around such agents, see [Observability](/haystack-guide/haystack_observability_python/).

---
title: "LlamaIndex: Advanced Implementation Patterns"
description: "Production-grade workflow patterns for LlamaIndex — parallel fan-out, agentic RAG with self-correction, retries, resilient tools and timing middleware."
framework: "llamaindex"
language: "python"
---

> Verified against `llama-index-core` %%version=llama-index%% (checked %%versions-checked%%). For multi-agent routing and human-in-the-loop see [Advanced agents](/llamaindex-guide/python/llamaindex_advanced_agents_python/).

Earlier versions of this page were built on the retired `llama-agents` package (`AgentOrchestrator`, `SimpleMessageQueue`) and on an `AgentWorkflow.from_tools(...)` constructor that never existed. The patterns below use the current Workflows runtime.

---

## 1. Parallel research with fan-out / fan-in

Run several agents concurrently and merge their output in a final step.

```python
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Workflow, step, Context, StartEvent, StopEvent, Event
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="%%model=openai-small%%")

def web_search(query: str) -> str:
    """Search the web."""
    return f"(results for {query})"

researcher = FunctionAgent(tools=[web_search], llm=llm,
                           system_prompt="Research the angle you are given. Reply with bullet notes.")

class Angle(Event):
    topic: str
    angle: str

class Notes(Event):
    angle: str
    notes: str

class ParallelResearch(Workflow):
    @step
    async def plan(self, ctx: Context, ev: StartEvent) -> Angle:
        angles = ["technical", "commercial", "regulatory"]
        await ctx.store.set("expected", len(angles))
        for a in angles:
            ctx.send_event(Angle(topic=ev.topic, angle=a))
        return None

    @step(num_workers=3)
    async def research(self, ev: Angle) -> Notes:
        result = await researcher.run(user_msg=f"{ev.topic} — {ev.angle} perspective")
        return Notes(angle=ev.angle, notes=str(result))

    @step
    async def merge(self, ctx: Context, ev: Notes) -> StopEvent | None:
        expected = await ctx.store.get("expected")
        batch = ctx.collect_events(ev, [Notes] * expected)
        if batch is None:
            return None
        combined = "\n\n".join(f"## {n.angle}\n{n.notes}" for n in batch)
        summary = await llm.acomplete(f"Write a one-paragraph synthesis:\n\n{combined}")
        return StopEvent(result=str(summary))

async def main():
    print(await ParallelResearch(timeout=300).run(topic="small modular reactors"))

asyncio.run(main())
```

`num_workers` caps concurrency for that step; `collect_events` returns `None` until all expected events have arrived.

---

## 2. Agentic RAG with self-correction

Retrieve, grade the context, and rewrite the query if the context is poor — bounded to a fixed number of attempts.

```python
from llama_index.core import VectorStoreIndex

class Retrieved(Event):
    query: str
    context: str
    attempt: int

class Rewrite(Event):
    query: str
    attempt: int

class CorrectiveRAG(Workflow):
    def __init__(self, index: VectorStoreIndex, max_attempts: int = 3, **kw):
        super().__init__(**kw)
        self.retriever = index.as_retriever(similarity_top_k=4)
        self.max_attempts = max_attempts

    @step
    async def start(self, ev: StartEvent) -> Rewrite:
        return Rewrite(query=ev.query, attempt=0)

    @step
    async def retrieve(self, ev: Rewrite) -> Retrieved:
        nodes = await self.retriever.aretrieve(ev.query)
        return Retrieved(query=ev.query, attempt=ev.attempt,
                         context="\n\n".join(n.get_content() for n in nodes))

    @step
    async def grade(self, ctx: Context, ev: Retrieved) -> Rewrite | StopEvent:
        verdict = await llm.acomplete(
            f"Does this context answer the question? Reply YES or NO.\n\nQ: {ev.query}\n\n{ev.context}"
        )
        if "YES" in str(verdict).upper() or ev.attempt + 1 >= self.max_attempts:
            answer = await llm.acomplete(f"Answer from the context only.\n\n{ev.context}\n\nQ: {ev.query}")
            return StopEvent(result=str(answer))
        better = await llm.acomplete(f"Rewrite this search query to be more specific: {ev.query}")
        return Rewrite(query=str(better).strip(), attempt=ev.attempt + 1)
```

The union return type (`Rewrite | StopEvent`) is how a step declares that it may branch; the runtime validates the graph from these annotations.

---

## 3. Step retries

```python
from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy

class Ingest(Workflow):
    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=4))
    async def fetch(self, ev: StartEvent) -> StopEvent:
        # any exception here re-runs the step after 5 s, up to 4 attempts
        ...
```

Use retry policies for idempotent steps only.

---

## 4. Resilient tools: retry, then fall back

```python
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from llama_index.core.tools import FunctionTool

def unreliable_api_call(query: str) -> str:
    if random.random() < 0.5:
        raise ConnectionError("upstream unavailable")
    return f"Data for {query}"

def fallback_search(query: str) -> str:
    return f"(cached results for {query})"

@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type(ConnectionError))
def _primary(query: str) -> str:
    return unreliable_api_call(query)

def resilient_search(query: str) -> str:
    """Search the primary API, falling back to the cache if it is unavailable."""
    try:
        return _primary(query)
    except ConnectionError:
        return fallback_search(query) + " [served from fallback]"

search_tool = FunctionTool.from_defaults(fn=resilient_search)
```

Telling the model that a result came from a fallback lets it caveat its answer.

---

## 5. Timing and logging middleware

Subclassing agents to wrap `run()` is fragile. The instrumentation dispatcher sees every LLM call, retrieval, tool call and workflow step without touching your classes:

```python
import logging
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent

log = logging.getLogger("llamaindex")

class LLMLogger(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        return "LLMLogger"

    def handle(self, event: BaseEvent, **kwargs) -> None:
        if isinstance(event, LLMChatEndEvent) and event.response is not None:
            usage = getattr(event.response.raw, "usage", None)
            log.info("llm call span=%s usage=%s", event.span_id, usage)

get_dispatcher().add_event_handler(LLMLogger())
```

For wall-clock timings and traces, export spans with OpenTelemetry — see the [observability page](/llamaindex-guide/python/llamaindex_observability_python/).

---

## 6. Caching ingestion

```python
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding

pipeline = IngestionPipeline(
    transformations=[SentenceSplitter(chunk_size=512), OpenAIEmbedding(model="text-embedding-3-small")],
    docstore=SimpleDocumentStore(),   # de-duplicates by document id + hash
    cache=IngestionCache(),           # skips re-running transformations on unchanged nodes
)
nodes = pipeline.run(documents=documents)
pipeline.persist("./pipeline_storage")   # pipeline.load("./pipeline_storage") next time
```

Use stable document IDs (`SimpleDirectoryReader(filename_as_id=True)`) so that changed files are upserted rather than duplicated. Remote cache and docstore backends (Redis, MongoDB, Postgres) are available as separate integration packages.

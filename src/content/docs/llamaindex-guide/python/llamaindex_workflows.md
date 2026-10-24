---
title: "LlamaIndex Workflows (Python)"
description: "The event-driven Workflows framework behind LlamaIndex agents, used on its own or with the rest of LlamaIndex: steps and events, typed state, fan-out, streaming, human-in-the-loop, retries, resources, checkpointing, testing."
framework: "llamaindex"
language: "python"
sidebar:
  label: "Workflows"
  order: 250
---

> Verified against `llama-index-workflows` %%version=llama-index-workflows%% and `llama-index-core` %%version=llama-index%% (checked %%versions-checked%%). Every example on this page was run against installed copies, with mock models where an LLM is involved.

Workflows is LlamaIndex's event-driven orchestration framework. A workflow is a class whose `@step` methods receive typed events and return new ones; the runtime routes each event to the steps that accept it, runs steps concurrently, and stops when a step returns a `StopEvent`. The prebuilt agents (`FunctionAgent`, `ReActAgent`, `CodeActAgent`, `AgentWorkflow`) are workflows themselves, and `WorkflowServer` serves any workflow over HTTP.

Workflows ship as a separate package, `llama-index-workflows`, imported as `workflows`. It has no dependency on the rest of LlamaIndex, so you can use it for any async pipeline, with or without LLMs.

---

## 1. Installation and imports

```bash
# Standalone: just the workflow engine
pip install llama-index-workflows==%%version=llama-index-workflows%%

# With LlamaIndex: llama-index-core depends on llama-index-workflows and re-exports it
pip install llama-index-core==%%version=llama-index%%
```

There are two import paths for the same classes:

| Standalone package | Through LlamaIndex |
|---|---|
| `from workflows import Workflow, step, Context, catch_error` | `from llama_index.core.workflow import Workflow, step, Context` |
| `from workflows.events import Event, StartEvent, StopEvent, InputRequiredEvent, HumanResponseEvent` | `from llama_index.core.workflow import Event, StartEvent, StopEvent, InputRequiredEvent, HumanResponseEvent` |
| `from workflows.retry_policy import retry_policy, ...` | `llama_index.core.workflow.retry_policy` only re-exports `RetryPolicy` and the deprecated `ConstantDelayRetryPolicy` |
| `from workflows.resource import Resource, ResourceConfig, ResourceManager` | `llama_index.core.workflow.resource` (`Resource` and `ResourceManager`, not `ResourceConfig`) |
| `from workflows.errors import WorkflowTimeoutError, ...` | `llama_index.core.workflow.errors` (all of them; the package root has the three most common) |

`llama_index.core.workflow.Workflow`, `Context` and the event classes are the same objects as their `workflows` counterparts, so you can mix the two paths. Newer features — `catch_error`, the composable retry policies, `ResourceConfig`, `workflows.collect`, `workflows.testing` — are only reachable through `workflows`. The examples below use the standalone imports.

---

## 2. A first workflow

```python
import asyncio
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent


class HelloFlow(Workflow):
    @step
    async def greet(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result=f"Hello, {ev.name}!")


async def main():
    wf = HelloFlow(timeout=10)
    result = await wf.run(name="Ada")
    print(result)  # Hello, Ada!

asyncio.run(main())
```

- Keyword arguments to `run()` become fields of the start event (`ev.name`). `StartEvent` accepts arbitrary fields; declare a typed start event (below) to validate them.
- `run()` returns a `WorkflowHandler` straight away. Awaiting it gives the `StopEvent`'s `result`; iterating `handler.stream_events()` gives intermediate events (section 6).
- `run()` must be called while an event loop is running, which is why it sits inside `async def main()`.
- Steps are `async def` methods. The type hints are not decoration: the runtime reads the event parameter to decide which events a step receives and the return annotation to validate the graph.

`Workflow(...)` takes `timeout` (seconds, default 45, `None` for no limit), `verbose` (print step activity), `disable_validation`, `num_concurrent_runs` (cap concurrent runs of this instance), `resource_manager` (section 10) and `workflow_name`.

---

## 3. Events, branching and loops

Custom events are Pydantic models that subclass `Event`. A step that may emit several kinds of event declares a union return type, and a step can send an event "back" to an earlier step to form a loop:

```python
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent


class Draft(Event):
    text: str
    attempt: int


class Review(Event):
    text: str
    attempt: int


class WriteAndReview(Workflow):
    @step
    async def start(self, ev: StartEvent) -> Draft:
        return Draft(text=ev.topic, attempt=1)

    @step
    async def write(self, ev: Draft) -> Review:
        return Review(text=f"{ev.text} (draft {ev.attempt})", attempt=ev.attempt)

    @step
    async def review(self, ev: Review) -> Draft | StopEvent:
        if ev.attempt < 3:
            return Draft(text=ev.text, attempt=ev.attempt + 1)  # loop back
        return StopEvent(result=ev.text)

# await WriteAndReview().run(topic="Workflows")
# -> 'Workflows (draft 1) (draft 2) (draft 3)'
```

Before the first run the workflow validates its graph: there must be exactly one start event type and one stop event type, every event a step returns must be consumed by some step, and every step must be reachable. A broken graph raises `WorkflowValidationError` (or `WorkflowConfigurationError` for start/stop problems) when you call `run()` or `validate()`.

A step may return `None` to emit nothing; add `| None` to the annotation when it does so on purpose.

### Typed start and stop events

Subclass `StartEvent` and `StopEvent` to give a workflow a typed interface. The runtime infers both from the step signatures:

```python
from pydantic import Field
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent


class SummariseStart(StartEvent):
    text: str
    max_words: int = Field(default=20, gt=0)


class SummariseResult(StopEvent):
    summary: str
    words: int


class Summarise(Workflow):
    @step
    async def run_it(self, ev: SummariseStart) -> SummariseResult:
        words = ev.text.split()[: ev.max_words]
        return SummariseResult(summary=" ".join(words), words=len(words))


async def main():
    wf = Summarise()
    r = await wf.run(text="one two three four five", max_words=3)
    print(r.summary, r.words)                 # one two three 3
    r = await wf.run(start_event=SummariseStart(text="a b c"))
    print(wf.start_event_class, wf.stop_event_class)
```

- `run(**kwargs)` builds the start event from keyword arguments and raises `WorkflowRuntimeError` if they do not validate; `run(start_event=...)` takes an instance.
- With a custom stop event, awaiting the handler returns **the stop event itself**, not a `result` attribute.
- `start_event_class` and `stop_event_class` expose the inferred types (used by `WorkflowServer` for its schema).

Events behave like dicts for undeclared fields (`ev.get("x")`, `ev["x"]`, `"x" in ev`). The flip side is that a dynamic field named `items`, `keys`, `values` or `get` is shadowed by the dict method: `run(items=[...])` followed by `ev.items` returns a bound method. Use another name or declare the field on a typed start event.

---

## 4. Context and state

A step that declares a `ctx: Context` parameter gets the run's context: a state store shared by all steps, plus the methods for sending, collecting and streaming events.

### Typed state

Declare the state as a Pydantic model and annotate the context as `Context[YourState]`. Edit it inside `ctx.store.edit_state()`, which takes a lock and commits the changes when the block exits:

```python
import asyncio
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent


class Visited(Event):
    page: str


class CrawlState(BaseModel):
    pages: list[str] = Field(default_factory=list)
    runs: int = 0


class Crawl(Workflow):
    @step
    async def start(self, ctx: Context[CrawlState], ev: StartEvent) -> Visited:
        async with ctx.store.edit_state() as state:
            state.runs += 1
        return Visited(page=ev.url)

    @step
    async def record(self, ctx: Context[CrawlState], ev: Visited) -> StopEvent:
        async with ctx.store.edit_state() as state:
            state.pages.append(ev.page)
        state = await ctx.store.get_state()
        return StopEvent(result={"runs": state.runs, "pages": state.pages})


async def main():
    wf = Crawl()
    ctx = Context(wf)
    print(await wf.run(ctx=ctx, url="/a"))   # {'runs': 1, 'pages': ['/a']}
    print(await wf.run(ctx=ctx, url="/b"))   # {'runs': 2, 'pages': ['/a', '/b']}
    print(await wf.run(url="/c"))            # fresh context: {'runs': 1, 'pages': ['/c']}

asyncio.run(main())
```

Passing the same `Context` to successive runs carries state from one run to the next; that is how agents keep a conversation going. Every model field needs a default, because the store creates the initial state by calling the model with no arguments. Between runs the state is serialised and rebuilt by class name, so define state models (and custom events) at module level: a model defined inside a function fails on the second run with `AttributeError: Attribute ... not found in module`.

How `edit_state()` behaves:

- It yields an isolated copy. Reads made inside the block (`get_state()`, `get()`) and reads from concurrent steps still see the committed state until the block exits.
- Calling a writer (`set`, `set_state`, `clear` or a nested `edit_state`) inside the block raises `RuntimeError`. Mutate the yielded object instead.
- Edits are serialised, so concurrent workers cannot lose each other's updates.

### Untyped state

Without a type parameter the store holds a `DictState`, read and written by key. Keys accept dot-separated paths:

```python
@step
async def go(self, ctx: Context, ev: StartEvent) -> StopEvent:
    await ctx.store.set("user", {"name": "Ada", "prefs": {"lang": "en"}})
    lang = await ctx.store.get("user.prefs.lang")          # 'en'
    missing = await ctx.store.get("nope", default=None)     # without default: raises
    return StopEvent(result=lang)
```

Keep state JSON-serialisable (or a Pydantic model) if you intend to checkpoint it (section 11). Clients, connections and other live objects belong in [resources](#10-resources-and-dependency-injection), not in state.

---

## 5. Concurrency: fan-out, fan-in and `num_workers`

Each step has a pool of workers, `num_workers` (default 4). Events queue for a step and up to `num_workers` of them are processed at once.

### `send_event` and `collect_events`

`ctx.send_event(ev)` emits an event without returning from the step, so one step can start many units of work. `ctx.collect_events(ev, expected)` buffers events and returns them once the expected set has arrived (and `None` until then):

```python
import asyncio
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent


class Job(Event):
    item: str


class Done(Event):
    item: str
    length: int


class FanOut(Workflow):
    @step
    async def split(self, ctx: Context, ev: StartEvent) -> Job | None:
        await ctx.store.set("expected", len(ev.inputs))
        for item in ev.inputs:
            ctx.send_event(Job(item=item))
        return None

    @step(num_workers=2)
    async def work(self, ev: Job) -> Done:
        await asyncio.sleep(0.2)                 # stands in for an API call
        return Done(item=ev.item, length=len(ev.item))

    @step
    async def join(self, ctx: Context, ev: Done) -> StopEvent | None:
        expected = await ctx.store.get("expected")
        batch = ctx.collect_events(ev, [Done] * expected)
        if batch is None:
            return None                          # still waiting for the rest
        return StopEvent(result={d.item: d.length for d in batch})

# await FanOut().run(inputs=["a", "bb", "ccc", "dddd"]) takes about 0.4 s with two workers
```

`collect_events` returns events in arrival order, not in the order they were sent. `ctx.send_event(ev, step="work")` targets a single step; it raises `WorkflowRuntimeError` if that step does not accept the event type.

### List returns and list parameters

A step can return `list[E]` to fan out, and a later step can take `list[E]` to receive the whole batch once every member has been handled. The runtime tracks the batch, so there is no counter to keep in state:

```python
from typing import Annotated
from workflows import Workflow, step
from workflows.collect import Collect, Take
from workflows.events import StartEvent, StopEvent


class ListFanOut(Workflow):
    @step
    async def split(self, ev: StartEvent) -> list[Job]:
        return [Job(item=i) for i in ev.inputs]

    @step
    async def work(self, ev: Job) -> Done:
        return Done(item=ev.item, length=len(ev.item))

    @step
    async def join(self, results: list[Done]) -> StopEvent:
        return StopEvent(result=sorted((d.item, d.length) for d in results))


class Fastest(Workflow):
    @step
    async def split(self, ev: StartEvent) -> list[Job]:
        return [Job(item=i) for i in ev.inputs]

    @step
    async def work(self, ev: Job) -> Done:
        await asyncio.sleep(0.05 * len(ev.item))
        return Done(item=ev.item, length=len(ev.item))

    @step
    async def first(self, results: Annotated[list[Done], Collect(Take(1))]) -> StopEvent:
        return StopEvent(result=results[0].item)   # the first Done to arrive
```

- A bare `list[Done]` parameter means `Collect(All())`: fire once, with every result, when the batch is complete. Returning `[]` still fires the join once, with an empty list.
- `Collect(Take(n))` fires on the n-th arrival; the remaining work keeps running but its results never reach this step.
- Events added with `ctx.send_event` do not join a list batch. Use one style or the other for a given fan-out.

### Waiting for different events

A step with several event parameters fires once one of each has arrived — a join across different event types:

```python
class Query(Event):
    text: str


class Retrieved(Event):
    chunks: list[str]


class Gather(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> Query | Retrieved:
        ctx.send_event(Query(text=ev.question))
        ctx.send_event(Retrieved(chunks=["c1", "c2"]))
        return None

    @step
    async def answer(self, query: Query, retrieved: Retrieved) -> StopEvent:
        return StopEvent(result=f"{query.text}: {retrieved.chunks}")
```

The older form — a step accepting `Query | Retrieved` that calls `ctx.collect_events(ev, [Query, Retrieved])` and unpacks the result — still works.

---

## 6. Streaming events

`ctx.write_event_to_stream(ev)` publishes an event to whoever is watching the run, without routing it to any step. The caller reads them from `handler.stream_events()`:

```python
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent


class Progress(Event):
    msg: str


class Token(Event):
    delta: str


class Report(Workflow):
    @step
    async def write(self, ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.write_event_to_stream(Progress(msg="drafting"))
        text = ""
        for word in ["Workflows", "stream", "events."]:   # stands in for LLM tokens
            text += word + " "
            ctx.write_event_to_stream(Token(delta=word + " "))
        return StopEvent(result=text.strip())


async def main():
    handler = Report().run()
    async for ev in handler.stream_events():
        if isinstance(ev, Progress):
            print(f"[{ev.msg}]")
        elif isinstance(ev, Token):
            print(ev.delta, end="", flush=True)
    print()
    print(await handler)
```

- The stream ends with the `StopEvent`. Awaiting the handler afterwards returns the result.
- A handler's stream can be consumed once; a second `stream_events()` call raises `WorkflowRuntimeError`.
- `stream_events(expose_internal=True)` also yields the runtime's own events, such as `StepStateChanged` (step name, `step_state` `running`/`not_running`, worker id, input and output event names) — useful for progress UIs and debugging.
- When a run ends abnormally the stream carries a `StopEvent` subclass that says why: `WorkflowTimedOutEvent` (`timeout`, `active_steps`), `WorkflowCancelledEvent` or `WorkflowFailedEvent` (`step_name`, `exception`, `attempts`). Awaiting the handler then raises the matching exception.

To stream over HTTP, iterate `stream_events()` in your endpoint (see the [FastAPI streaming page](/llamaindex-guide/python/llamaindex_streaming_server_fastapi/)) or let [`WorkflowServer`](#13-serving) do it.

---

## 7. Human-in-the-loop

Two built-in events cover pausing for a person: `InputRequiredEvent`, which a workflow emits to the stream, and `HumanResponseEvent`, which the caller sends back. Both accept arbitrary fields and can be subclassed.

### As an event round-trip

A step that returns an `InputRequiredEvent` writes it to the stream and the run waits. A step that accepts a `HumanResponseEvent` resumes the flow when the caller sends one:

```python
from workflows import Workflow, step
from workflows.events import HumanResponseEvent, InputRequiredEvent, StartEvent, StopEvent


class AskApproval(InputRequiredEvent):
    question: str


class Approval(HumanResponseEvent):
    approved: bool


class Publish(Workflow):
    @step
    async def draft(self, ev: StartEvent) -> AskApproval:
        return AskApproval(question=f"Publish '{ev.title}'?")

    @step
    async def decide(self, ev: Approval) -> StopEvent:
        return StopEvent(result="published" if ev.approved else "discarded")


async def main():
    handler = Publish().run(title="Release notes")
    async for ev in handler.stream_events():
        if isinstance(ev, AskApproval):
            answer = input(ev.question + " ")          # or a web form, a Slack button...
            await handler.send_event(Approval(approved=answer == "yes"))
    print(await handler)
```

`InputRequiredEvent` is exempt from the "every event must be consumed" check and `HumanResponseEvent` from the "every step must be reachable" check, so this graph validates. `Workflow.validate()` returns `True` for workflows that use them.

`handler.send_event(...)` is a coroutine. `handler.ctx.send_event(...)` does the same thing synchronously.

### Inside a step, with `wait_for_event`

`ctx.wait_for_event` pauses the current step until a matching event arrives:

```python
@step
async def refund(self, ctx: Context, ev: StartEvent) -> StopEvent:
    answer = await ctx.wait_for_event(
        HumanResponseEvent,
        waiter_event=InputRequiredEvent(prefix=f"Refund {ev.amount}? "),
        waiter_id=f"refund-{ev.order_id}",
        requirements={"order_id": ev.order_id},
        timeout=300,
    )
    if answer.response != "yes":
        return StopEvent(result="refund declined")
    return StopEvent(result=f"refunded {ev.amount}")

# caller: handler.ctx.send_event(HumanResponseEvent(response="yes", order_id="A1"))
```

- `waiter_event` is written to the stream once per `waiter_id`.
- `requirements` filters on event fields; a `HumanResponseEvent` with a different `order_id` is ignored.
- `timeout` defaults to 2000 seconds; `None` waits indefinitely (still bounded by the workflow's own `timeout` — set that to `None` for long waits).
- **The step runs again from the top** when the event arrives: the runtime pauses by raising an internal exception and replays the step. Put `wait_for_event` at the start of the step and make anything before it safe to repeat. In the example above the step body executes twice.

The same call works inside a tool given to a prebuilt agent; see [§15 of the comprehensive guide](/llamaindex-guide/python/llamaindex_comprehensive_guide/#15-human-in-the-loop). For an approval that must survive a restart, checkpoint the context (section 11).

---

## 8. Errors, retries and timeouts

### Retry policies

`@step(retry_policy=...)` re-runs a step that raised. Build policies with `retry_policy()` from three parts, the same vocabulary as `tenacity`: what to retry, how long to wait, when to stop.

```python
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.retry_policy import (
    retry_policy, retry_if_exception_type, stop_after_attempt, wait_exponential,
)


class Flaky(Workflow):
    @step(
        retry_policy=retry_policy(
            retry=retry_if_exception_type((ConnectionError, TimeoutError)),
            wait=wait_exponential(multiplier=0.5, max=10),
            stop=stop_after_attempt(4),
        )
    )
    async def fetch(self, ctx: Context, ev: StartEvent) -> StopEvent:
        info = ctx.retry_info()        # retry_number, elapsed_seconds, last_exception, last_failed_at
        if info.last_exception is not None:
            print(f"retry {info.retry_number} after: {info.last_exception}")
        return StopEvent(result=await call_upstream())
```

- With no arguments, `retry_policy()` retries any exception up to 3 attempts with a fixed 5-second wait.
- Retry conditions: `retry_if_exception_type`, `retry_if_not_exception_type`, `retry_if_exception_message`, `retry_if_exception_cause_type`, `retry_if_exception`, combined with `retry_any` / `retry_all`.
- Waits: `wait_fixed`, `wait_none`, `wait_exponential`, `wait_exponential_jitter`, `wait_random_exponential`, `wait_random`, `wait_incrementing`, `wait_chain`, `wait_combine`.
- Stops: `stop_after_attempt`, `stop_after_delay`, `stop_before_delay`, `stop_any`, `stop_all`, `stop_never`.
- `ConstantDelayRetryPolicy` and `ExponentialBackoffRetryPolicy` still exist but emit a `DeprecationWarning`. The equivalent of `ConstantDelayRetryPolicy(delay=5, maximum_attempts=3)` is `retry_policy(wait=wait_fixed(5), stop=stop_after_attempt(3))`.

The whole step is retried, so make retried steps idempotent.

### Recovering with `@catch_error`

When a step has exhausted its retries, a `@catch_error` handler can take over instead of failing the run. It receives a `StepFailedEvent` (`step_name`, `input_event`, `exception`, `attempts`, `elapsed_seconds`, `failed_at`) and may return any event — a `StopEvent` to finish, or an event that rejoins the normal flow:

```python
from workflows import Workflow, catch_error, step
from workflows.events import Event, StartEvent, StepFailedEvent, StopEvent
from workflows.retry_policy import retry_policy, stop_after_attempt, wait_fixed


class Fetched(Event):
    body: str


class WithFallback(Workflow):
    @step(retry_policy=retry_policy(wait=wait_fixed(1), stop=stop_after_attempt(2)))
    async def fetch(self, ev: StartEvent) -> Fetched:
        raise ConnectionError("upstream down")

    @step
    async def parse(self, ev: Fetched) -> StopEvent:
        return StopEvent(result=ev.body)

    @catch_error(for_steps=["fetch"])
    async def fallback(self, ev: StepFailedEvent) -> Fetched:
        return Fetched(body=f"cached copy ({ev.step_name} failed {ev.attempts}x: {ev.exception})")

# await WithFallback().run() -> 'cached copy (fetch failed 2x: upstream down)'
```

A bare `@catch_error` covers every step not claimed by a scoped handler. `max_recoveries` (default 1) limits how often a handler can be re-entered for the same chain of events before the run fails. Without a handler, awaiting the run re-raises the step's exception.

### Timeouts and cancellation

```python
from workflows.errors import WorkflowTimeoutError

try:
    await SlowFlow(timeout=30).run()
except WorkflowTimeoutError as e:
    print(e)    # "Operation timed out after ... seconds", plus the steps still active

handler = SlowFlow(timeout=None).run()
...
await handler.cancel_run()   # awaiting the handler then raises WorkflowCancelledByUser
```

The timeout covers the whole run, including time spent waiting for a human. `handler.cancel()`, `handler.done()`, `handler.exception()` and `handler.cancelled()` are deprecated; use `await handler.cancel_run()` and `handler.is_done()`.

---

## 9. Calling LLMs and other LlamaIndex components

Nothing in a step is special: call an LLM, a retriever or a query engine as you would anywhere else.

```python
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI


class RetrievedContext(Event):
    query: str
    context: str


class RAGWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex, **kwargs):
        super().__init__(**kwargs)
        self.retriever = index.as_retriever(similarity_top_k=3)
        self.llm = OpenAI(model="%%model=openai-small%%")

    @step
    async def retrieve(self, ev: StartEvent) -> RetrievedContext:
        nodes = await self.retriever.aretrieve(ev.query)
        return RetrievedContext(query=ev.query, context="\n\n".join(n.get_content() for n in nodes))

    @step
    async def synthesise(self, ctx: Context, ev: RetrievedContext) -> StopEvent:
        prompt = f"Answer using only this context.\n\n{ev.context}\n\nQuestion: {ev.query}"
        text = ""
        async for chunk in await self.llm.astream_complete(prompt):
            ctx.write_event_to_stream(Token(delta=chunk.delta))
            text += chunk.delta
        return StopEvent(result=text)
```

Passing dependencies to `__init__` is fine for one-off scripts. For anything you want to swap in tests or share between workflows, use resources.

---

## 10. Resources and dependency injection

A step parameter annotated with `Annotated[T, Resource(factory)]` is filled in by the runtime. By default the factory runs once and the instance is shared by every step that asks for it (`cache=True`); `Resource(factory, cache=False)` builds a new one per step call. Factories may be `async` and may themselves take resource parameters.

```python
from typing import Annotated
from pydantic import BaseModel
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig


class HttpClient:
    async def get(self, url: str) -> str:
        return f"<body of {url}>"


def get_client() -> HttpClient:
    return HttpClient()


class ClassifierConfig(BaseModel):
    labels: list[str]
    threshold: float


class Page(Event):
    body: str


class Triage(Workflow):
    @step
    async def fetch(
        self, ev: StartEvent, client: Annotated[HttpClient, Resource(get_client)]
    ) -> Page:
        return Page(body=await client.get(ev.url))

    @step
    async def classify(
        self,
        ev: Page,
        config: Annotated[
            ClassifierConfig,
            ResourceConfig(config_file="classifier.json", path_selector="classifier"),
        ],
    ) -> StopEvent:
        return StopEvent(result=(ev.body, config.labels))
```

`ResourceConfig` loads a JSON file (here `{"classifier": {"labels": [...], "threshold": 0.5}}`) into a Pydantic model; `path_selector` picks a nested key. Config files are checked when the workflow validates.

To replace a resource — a fake client in a test, say — register an instance under the factory's qualified name in a `ResourceManager` and pass it to the workflow:

```python
from workflows.resource import ResourceManager

rm = ResourceManager()
await rm.set("get_client", FakeClient())
result = await Triage(resource_manager=rm).run(url="https://example.com")
```

Resources are not part of the serialised context, which is the point: they are rebuilt wherever the run resumes.

---

## 11. Checkpointing and resuming

`handler.ctx.to_dict()` serialises a run — state, queued events, in-flight work and whether it is running — to a JSON-compatible dict. `Context.from_dict(workflow, data)` rebuilds it, and `workflow.run(ctx=...)` continues the run rather than starting a new one. Together with human-in-the-loop this gives approval steps that survive a restart:

```python
import json
from pydantic import BaseModel
from workflows import Context, Workflow, step
from workflows.events import HumanResponseEvent, InputRequiredEvent, StartEvent, StopEvent


class AskApproval(InputRequiredEvent):
    question: str


class Approval(HumanResponseEvent):
    approved: bool


class Notes(BaseModel):
    title: str = ""


class Publish(Workflow):
    @step
    async def draft(self, ctx: Context[Notes], ev: StartEvent) -> AskApproval:
        async with ctx.store.edit_state() as s:
            s.title = ev.title
        return AskApproval(question=f"Publish '{ev.title}'?")

    @step
    async def decide(self, ctx: Context[Notes], ev: Approval) -> StopEvent:
        title = (await ctx.store.get_state()).title
        return StopEvent(result=f"{title}: {'published' if ev.approved else 'discarded'}")


async def start_and_park() -> str:
    handler = Publish().run(title="Release notes")
    async for ev in handler.stream_events():
        if isinstance(ev, AskApproval):
            break
    saved = json.dumps(handler.ctx.to_dict())   # store this in a database
    await handler.cancel_run()                  # this process stops waiting
    return saved


async def resume(saved: str) -> None:
    wf = Publish()                              # a fresh instance, e.g. in another process
    ctx = Context.from_dict(wf, json.loads(saved))
    handler = wf.run(ctx=ctx)                   # no start event: the run continues
    await handler.send_event(Approval(approved=True))
    print(await handler)                        # Release notes: published
```

- Serialise after the run has reached a stable point, such as an `InputRequiredEvent`.
- The default `JsonSerializer` handles JSON values, Pydantic models (stored with their class name) and LlamaIndex components. `PickleSerializer` (`from workflows.context import PickleSerializer`, formerly `JsonPickleSerializer`) falls back to pickle; only load pickled contexts you produced yourself. Pass the serializer to both `to_dict(serializer=...)` and `from_dict(..., serializer=...)`.
- `to_dict(include_state=False)` omits the state store.
- A prebuilt agent paused inside a tool cannot currently be resumed this way, because its memory is not serialised with the context. The [durable approval pattern](/llamaindex-guide/python/llamaindex_advanced_agents_python/#42-approval-that-outlives-the-process) keeps the paused part in a small custom workflow.

`WorkflowServer` with a SQLite store does this persistence for you (section 13).

---

## 12. Nested workflows

A step can run another workflow and await it like any other coroutine. Pass the child in as a resource so it can be replaced in tests, and forward the child's stream if callers should see its progress:

```python
from typing import Annotated
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource


class Progress(Event):
    msg: str


class Summarise(Workflow):
    @step
    async def summarise(self, ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.write_event_to_stream(Progress(msg=f"summarising {len(ev.text)} chars"))
        return StopEvent(result=ev.text[:200])


def get_summariser() -> Summarise:
    return Summarise(timeout=30)


class Pipeline(Workflow):
    @step
    async def run_child(
        self,
        ctx: Context,
        ev: StartEvent,
        summariser: Annotated[Summarise, Resource(get_summariser)],
    ) -> StopEvent:
        handler = summariser.run(text=ev.text)
        async for child_ev in handler.stream_events():
            if isinstance(child_ev, Progress):
                ctx.write_event_to_stream(child_ev)    # forward to the parent's stream
        summary = await handler
        return StopEvent(result=f"summary: {summary}")
```

The child has its own context, state and timeout. Agents nest the same way: `await agent.run(user_msg=...)` inside a step (section 15).

---

## 13. Serving

A workflow is an async object, so any web framework can run it. LlamaIndex's own serving layer — `WorkflowServer`, `WorkflowClient` and the `llamactl` CLI, with durable run storage and event streaming over HTTP — is covered in [§14 of the comprehensive guide](/llamaindex-guide/python/llamaindex_comprehensive_guide/#14-serving-and-deploying-workflows-llamaagents) and the [production guide](/llamaindex-guide/python/llamaindex_production_guide/#4-serving-agents-and-workflows-with-workflowserver).

---

## 14. Drawing and testing

### Drawing the graph

```bash
pip install llama-index-utils-workflow
```

```python
from llama_index.utils.workflow import draw_all_possible_flows, draw_all_possible_flows_mermaid

draw_all_possible_flows(WriteAndReview, filename="write_review.html")      # interactive HTML (pyvis)
mermaid = draw_all_possible_flows_mermaid(WriteAndReview, filename="write_review.mermaid")
```

Both accept a class or an instance. With `include_child_workflows=True` (the default) a workflow instantiated directly in a step body is drawn as a nested graph; one supplied as a resource is not. `draw_most_recent_execution(handler)` draws the path a finished run actually took, and `draw_agent_workflow` / `draw_agent_with_tools` draw agents. `llama_index.core.workflow.drawing.draw_all_possible_flows` is deprecated in favour of this package.

For your own tooling, `workflows.representation.get_workflow_representation(MyWorkflow)` returns the graph as a Pydantic model (`nodes`, `edges`) without any drawing dependency.

### Testing

`WorkflowTestRunner` runs a workflow and collects what it streamed:

```python
import pytest
from workflows.events import StartEvent, StopEvent
from workflows.testing import WorkflowTestRunner


@pytest.mark.asyncio
async def test_report_streams_tokens():
    result = await WorkflowTestRunner(Report()).run(start_event=StartEvent(), expose_internal=False)
    assert result.result == "Workflows stream events."
    assert result.event_types[Token] == 3          # counts per event class


@pytest.mark.asyncio
async def test_review_stops_on_third_attempt():
    wf = WriteAndReview()
    out = await wf.review(Review(text="x", attempt=3))   # a step without ctx is a plain coroutine
    assert isinstance(out, StopEvent)


def test_graph_is_valid():
    WriteAndReview().validate()                    # raises WorkflowValidationError if broken
```

- `WorkflowTestRunner.run()` takes `start_event` (an instance), `ctx`, `expose_internal` (default `True`, which includes `StepStateChanged` events) and `exclude_events`. The result has `collected`, `event_types`, `result` and `ctx`.
- Use mock models for anything that calls an LLM: `MockLLM` echoes the prompt, `MockFunctionCallingLLM` drives agent tool calls ([§5.4](/llamaindex-guide/python/llamaindex_comprehensive_guide/#54-offline-testing-with-mock-models)). Inject fakes for external services through resources.
- The async tests above need `pytest-asyncio`.

---

## 15. Agents are workflows

`FunctionAgent`, `ReActAgent` and `CodeActAgent` subclass `BaseWorkflowAgent`, which subclasses `Workflow`; `AgentWorkflow` is a `Workflow` too. A single agent's graph looks like this:

| Step | Receives | Emits |
|---|---|---|
| `init_run` | `AgentWorkflowStartEvent` (`user_msg`, `chat_history`, `max_iterations`) | `AgentInput` |
| `setup_agent` | `AgentInput` | `AgentSetup` |
| `run_agent_step` | `AgentSetup` | `AgentOutput` (streams `AgentStream` tokens) |
| `parse_agent_output` | `AgentOutput` | `ToolCall` events; `StopEvent` when the model calls no tools; `AgentInput` to retry |
| `call_tool` | `ToolCall` | `ToolCallResult` |
| `aggregate_tool_results` | `ToolCallResult` | `AgentInput` (loop) or `StopEvent` |

So everything on this page applies to agents: `agent.run(...)` returns a `WorkflowHandler`, `stream_events()` yields `AgentInput`, `AgentStream`, `AgentOutput`, `ToolCall` and `ToolCallResult`, a shared `Context(agent)` carries memory between runs, tools can take `ctx: Context` to read and write state, and tools can call `ctx.wait_for_event` for approval. `AgentWorkflow` adds hand-off between several agents on the same machinery. The agent classes are documented in [§4](/llamaindex-guide/python/llamaindex_comprehensive_guide/#4-agent-classes-and-types) and [§12](/llamaindex-guide/python/llamaindex_comprehensive_guide/#12-multi-agent-systems-with-agentworkflow) of the comprehensive guide.

Going the other way, an agent can be one step of a larger workflow, which is how you combine fixed control flow with model-driven parts:

```python
from llama_index.core.agent.workflow import FunctionAgent, ToolCall
from llama_index.llms.openai import OpenAI


def lookup_order(order_id: str) -> str:
    """Look up an order's status."""
    return f"order {order_id}: shipped"


agent = FunctionAgent(tools=[lookup_order], llm=OpenAI(model="%%model=openai-small%%"),
                      system_prompt="You help with orders.")


class Classified(Event):
    question: str


class SupportDesk(Workflow):
    @step
    async def classify(self, ev: StartEvent) -> Classified | StopEvent:
        if "order" not in ev.question.lower():
            return StopEvent(result="Please contact sales.")
        return Classified(question=ev.question)

    @step
    async def answer(self, ctx: Context, ev: Classified) -> StopEvent:
        handler = agent.run(user_msg=ev.question)
        async for agent_ev in handler.stream_events():
            if isinstance(agent_ev, ToolCall):
                ctx.write_event_to_stream(agent_ev)     # surface the agent's tool calls
        return StopEvent(result=str(await handler))
```

[Workflow patterns](/llamaindex-guide/python/llamaindex_advanced_implementations/) has longer examples in this style: parallel research agents, corrective RAG, resilient tools.

---

## 16. Common pitfalls

| Symptom | Cause and fix |
|---|---|
| `WorkflowValidationError` on the first `run()` | An event is produced but no step accepts it, a step is unreachable, or a return annotation is missing. Annotate every step's parameter and return type; return `None` with `\| None` in the annotation. |
| `RuntimeError: no running event loop` | `run()` was called outside a coroutine. Call it inside `async def main()` and `asyncio.run(main())`. |
| `ev.items` is a method | Dynamic event fields called `items`, `keys`, `values` or `get` are shadowed. Rename the field or declare it on a typed start event. |
| Work before `wait_for_event` happens twice | The step is replayed when the awaited event arrives. Move the call to the top of the step. |
| Run ends with `WorkflowTimeoutError` while waiting for a person | The default workflow timeout is 45 seconds. Use `timeout=None` (or a long value) for workflows that wait on humans. |
| `RuntimeError` inside `edit_state()` | A writer (`set`, `set_state`) was called inside the block. Mutate the yielded state object. |
| Fan-in never fires | `collect_events` was given the wrong count, or events were sent with `send_event` to a `list[E]` join. Store the expected count before sending, or use `list[E]` returns. |
| `DeprecationWarning: ConstantDelayRetryPolicy is deprecated` | Use `retry_policy(wait=wait_fixed(...), stop=stop_after_attempt(...))`. |
| Second `stream_events()` raises | A handler's stream can be read once. Collect events in the first loop. |
| `AttributeError: Attribute X not found in module` on a second run or a resume | The state model or an event class was defined inside a function. Move it to module level so it can be imported by name. |

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | llama-index-workflows 2.26.1 | New page: Workflows as a framework in its own right, replacing the short custom-workflow section of the comprehensive guide. Covers list fan-out and `Collect`, multi-event joins, `@catch_error`, composable retry policies, resources, checkpointing, nested workflows, drawing, testing and the agent step graph. All examples run against the installed packages. |

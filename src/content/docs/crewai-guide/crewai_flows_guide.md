---
title: "CrewAI Flows"
description: "Event-driven workflows with crewai.flow — structured state, @start/@listen/@router, and_/or_, @persist, human feedback, conversational and declarative flows."
framework: "crewai"
language: "python"
sidebar:
  label: "Flows"
  order: 30
---

> **Verified against `crewai` %%version=crewai%%** (checked %%versions-checked%%; source: `crewai/flow/dsl/`, `crewai/flow/runtime/`, `crewai/flow/persistence/`, `crewai/flow/human_feedback.py`, `crewai/flow/conversational*.py`). Every example on this page was run against that release with a stub LLM.

A **Flow** is a Python class whose methods are wired together by decorators: `@start()` methods run at kickoff, `@listen(...)` methods run when something they listen to finishes, and `@router(...)` methods return a label that decides which listeners run next. Flows hold typed state, can call crews or single agents from any method, and can be persisted, paused for human review and resumed.

Use a crew when a team of agents should work through a list of tasks. Use a flow when you need control around that: branching, loops, plain Python steps between LLM calls, several crews in sequence, or a run that has to survive a restart.

In the 1.15 line `Flow` is a Pydantic model. The decorators build a `FlowDefinition` (the same structure a YAML flow produces), and it is validated when you instantiate the class, so wiring mistakes fail before any method runs.

## A first flow

```python
from pydantic import BaseModel
from crewai.flow import Flow, start, listen


class GreetingState(BaseModel):
    name: str = "world"
    greeting: str = ""


class GreetingFlow(Flow[GreetingState]):
    @start()
    def build_greeting(self):
        self.state.greeting = f"Hello, {self.state.name}"
        return self.state.greeting

    @listen(build_greeting)
    def shout(self, greeting: str):
        return greeting.upper()


flow = GreetingFlow()
print(flow.kickoff(inputs={"name": "Ada"}))   # HELLO, ADA
print(flow.state.id)                           # a UUID, generated for every run
```

- `Flow[GreetingState]` gives the flow a **structured state**. `kickoff(inputs=...)` sets matching fields before the first method runs; fields need defaults.
- A listener can take the previous method's return value as a parameter (`greeting` above), or read `self.state`.
- `kickoff()` returns the output of the last method that ran. When two listeners fire from the same trigger, which output comes back is not something to rely on (in a test where two listeners shared one trigger, it was the first one declared); put results you need in state.
- Every state gets an `id` field (a UUID) even if your model does not declare one. Persistence uses it.

`self.state` is a read-only property: assign to its fields (`self.state.greeting = ...`), never to `self.state` itself. Overriding `__init__` to set `self.state = {...}` raises `AttributeError: property 'state' ... has no setter`.

### Unstructured state

Leave out the type parameter and `self.state` is a dict:

```python
from crewai.flow import Flow, start, listen


class TallyFlow(Flow):                      # no type parameter: state is a dict
    @start()
    def first(self):
        self.state["items"] = [self.state.get("seed", "a")]

    @listen(first)
    def second(self):
        self.state["items"].append("b")
        return self.state["items"]


print(TallyFlow().kickoff(inputs={"seed": "z"}))   # ['z', 'b']
```

Kickoff inputs are merged into the dict. Structured state is the better default: typos become validation errors, and persisted state round-trips cleanly.

## Decorators

| Decorator | Runs when | Notes |
|---|---|---|
| `@start()` | At kickoff | A flow can have **one or more** start methods; they all run. |
| `@start("label")` / `@start(method)` | At kickoff, and again when the label or method fires | Useful for retry loops. |
| `@listen(method)` | After `method` finishes | Receives that method's return value if it takes a parameter. |
| `@listen("label")` | When a router returns `"label"` (or a method named `label` finishes) | |
| `@listen(and_(a, b))` | After **all** of `a`, `b` have finished | Fan-in. |
| `@listen(or_(a, b))` | After the **first** of `a`, `b` finishes | |
| `@router(method)` | After `method` finishes | Its return value is a label; listeners on that label run next. |

`@listen` takes exactly one condition. `@listen(a, b, c)` raises `TypeError: listen() takes 1 positional argument`; combine triggers with `and_` or `or_`.

### Routers

```python
from typing import Literal
from pydantic import BaseModel
from crewai.flow import Flow, start, listen, router


class TicketState(BaseModel):
    text: str = ""
    priority: str = ""


class TriageFlow(Flow[TicketState]):
    @start()
    def classify(self):
        self.state.priority = "high" if "outage" in self.state.text.lower() else "low"

    @router(classify)
    def route(self) -> Literal["urgent", "routine"]:
        return "urgent" if self.state.priority == "high" else "routine"

    @listen("urgent")
    def page_on_call(self):
        return f"Paged on-call: {self.state.text}"

    @listen("routine")
    def add_to_queue(self):
        return f"Queued: {self.state.text}"


print(TriageFlow().kickoff(inputs={"text": "Checkout outage in EU"}))   # Paged on-call: ...
print(TriageFlow().kickoff(inputs={"text": "Typo on pricing page"}))    # Queued: ...
```

- The `Literal[...]` return annotation declares the router's labels for the flow definition and `plot()`. Without one, pass them explicitly: `@router(classify, emit=["urgent", "routine"])`. Routing works either way.
- **A route label must not be the name of the method that listens to it.** `return "page_on_call"` with `@listen("page_on_call") def page_on_call(...)` is rejected when the flow is instantiated: `Invalid flow definition ... listen condition 'page_on_call' references the handler name 'page_on_call'. A listener triggered by its own completion creates an infinite loop.` Use labels that describe the outcome (`"urgent"`) and method names that describe the action (`page_on_call`).

### Fan-out and fan-in with `and_` / `or_`

```python
from pydantic import BaseModel
from crewai.flow import Flow, start, listen, and_, or_


class ReportState(BaseModel):
    topic: str = ""
    news: str = ""
    filings: str = ""


class ReportFlow(Flow[ReportState]):
    @start()
    def fetch_news(self):
        self.state.news = f"news about {self.state.topic}"

    @start()
    def fetch_filings(self):
        self.state.filings = f"filings for {self.state.topic}"

    @listen(and_(fetch_news, fetch_filings))     # waits for both
    def write_report(self):
        return f"Report: {self.state.news} + {self.state.filings}"

    @listen(or_(fetch_news, fetch_filings))      # fires on the first to finish
    def log_progress(self):
        print("first source in")


print(ReportFlow().kickoff(inputs={"topic": "ACME"}))
```

Both start methods run at kickoff. Conditions nest: `and_(a, or_(b, c))`.

### Loops

A start method with a condition runs at kickoff and again each time the condition fires, which makes a bounded retry loop straightforward:

```python
from typing import Literal
from pydantic import BaseModel
from crewai.flow import Flow, start, listen, router


class DraftState(BaseModel):
    draft: str = ""
    attempts: int = 0


class RetryFlow(Flow[DraftState]):
    @start("retry")                 # runs at kickoff, and again whenever "retry" is emitted
    def write(self):
        self.state.attempts += 1
        self.state.draft = "x" * (40 * self.state.attempts)

    @router(write)
    def check(self) -> Literal["done", "give_up", "retry"]:
        if len(self.state.draft) >= 100:
            return "done"
        if self.state.attempts >= 5:
            return "give_up"
        return "retry"

    @listen("done")
    def finish(self):
        return f"accepted after {self.state.attempts} attempts"

    @listen("give_up")
    def escalate(self):
        return "sent to a human"


print(RetryFlow().kickoff())   # accepted after 3 attempts
```

Always keep a counter in state and an exit label. As a backstop, `Flow.max_method_calls` (default `100`) makes a run fail with `RecursionError: Method 'check' has been called 100 times in this flow execution, which indicates an infinite loop` when one method is invoked that often; a subclass can lower it with `max_method_calls: int = 10`. Treat it as a safety net, not as your loop condition.

## Calling crews and agents

Any flow method can run a crew or a single agent. Keep the crew definition in a function (or a `@CrewBase` class) so each run gets fresh objects:

```python
from pydantic import BaseModel
from crewai import Agent, Crew, Task, LLM
from crewai.flow import Flow, start, listen

llm = LLM(model="openai/%%model=openai-small%%")


class Headline(BaseModel):
    text: str


class ArticleState(BaseModel):
    topic: str = ""
    research: str = ""
    headline: str = ""


def research_crew() -> Crew:
    researcher = Agent(
        role="Researcher",
        goal="Collect accurate facts about {topic}",
        backstory="Fact-checker at a technical publication.",
        llm=llm,
    )
    task = Task(
        description="List five facts about {topic}, each with a source.",
        expected_output="Five bullet points with sources.",
        agent=researcher,
    )
    return Crew(agents=[researcher], tasks=[task])


class ArticleFlow(Flow[ArticleState]):
    @start()
    def research(self):
        result = research_crew().kickoff(inputs={"topic": self.state.topic})
        self.state.research = result.raw

    @listen(research)
    def write_headline(self):
        editor = Agent(
            role="Editor",
            goal="Write one accurate headline",
            backstory="Sub-editor who hates clickbait.",
            llm=llm,
        )
        out = editor.kickoff(
            f"Write a headline for these notes:\n{self.state.research}",
            response_format=Headline,
        )
        self.state.headline = out.pydantic.text
        return self.state.headline


print(ArticleFlow().kickoff(inputs={"topic": "solid-state batteries"}))
```

`Agent.kickoff(messages, response_format=...)` runs one agent without a crew and returns a `LiteAgentOutput` (`.raw`, `.pydantic`). In an `async def` flow method use `await crew.akickoff(...)` or `await agent.kickoff_async(...)`.

`crewai create flow <name>` scaffolds a project with a `crews/` folder, and `crewai flow add-crew <name>` adds another crew to it (see [CLI](/crewai-guide/crewai_cli_python/#flows)).

## Persistence and resume — `@persist`

`@persist` saves the state after each method completes. On a class it applies to every method; on a method, only to that one. Without an argument it uses the registered default backend (`flow_states.db` in CrewAI's per-project data directory, which is named after the current directory unless `CREWAI_STORAGE_DIR` is set); pass a `SQLiteFlowPersistence(path)` to choose the file.

```python
from pydantic import BaseModel
from crewai.flow import Flow, start, persist
from crewai.flow.persistence import SQLiteFlowPersistence


class CounterState(BaseModel):
    runs: int = 0


@persist(SQLiteFlowPersistence("flow_state.db"))
class CounterFlow(Flow[CounterState]):
    @start()
    def bump(self):
        self.state.runs += 1
        return self.state.runs


first = CounterFlow()
first.kickoff()
run_id = first.state.id

again = CounterFlow()
print(again.kickoff(inputs={"id": run_id}))           # 2: state restored, then bumped

fork = CounterFlow()
print(fork.kickoff(restore_from_state_id=run_id))     # 3: copy of that state under a new id
```

- `kickoff(inputs={"id": ...})` loads the latest saved state for that id and keeps writing under it.
- `kickoff(restore_from_state_id=...)` starts from a copy of that state with a fresh `state.id`, so the original history is untouched. It cannot be combined with `from_checkpoint`.
- For backends other than SQLite, subclass `crewai.flow.persistence.FlowPersistence` (`init_db`, `save_state`, `load_state`, plus the pending-feedback methods if you use asynchronous review).

Persistence is about **state**. Checkpoints (`kickoff(from_checkpoint=CheckpointConfig(restore_from=...))`) snapshot the whole run, crews included; see [Production](/crewai-guide/crewai_production_guide/#recovery-with-checkpoints).

## Human feedback — `@human_feedback`

Stack `@human_feedback` under `@start()` or `@listen(...)`. When the method returns, the flow shows its output, collects feedback, and (when `emit` is given) uses an LLM to map the free-text answer to one of the labels:

```python
from pydantic import BaseModel
from crewai import LLM
from crewai.flow import Flow, start, listen, human_feedback, HumanFeedbackResult


class DraftState(BaseModel):
    topic: str = ""
    draft: str = ""


class ReviewFlow(Flow[DraftState]):
    @start()
    @human_feedback(
        message="Approve this draft, or say what to change:",
        emit=["approved", "needs_changes"],
        llm=LLM(model="openai/%%model=openai-small%%"),   # maps free-text feedback to one of emit
        default_outcome="approved",                         # used when the reviewer just presses Enter
    )
    def write_draft(self):
        self.state.draft = f"Draft post about {self.state.topic}"
        return self.state.draft

    @listen("approved")
    def publish(self, result: HumanFeedbackResult):
        return f"Published: {result.output}"

    @listen("needs_changes")
    def revise(self, result: HumanFeedbackResult):
        return f"Revising because: {result.feedback}"


print(ReviewFlow().kickoff(inputs={"topic": "release notes"}))
```

- The listener receives a `HumanFeedbackResult` (`output`, `feedback`, `outcome`, `timestamp`, `method_name`, `metadata`). `self.last_human_feedback` and `self.human_feedback_history` hold the same data.
- `default_outcome` must be one of `emit`; it is used for empty feedback.
- Without `emit` the feedback does not route: ordinary `@listen(write_draft)` listeners run and receive the `HumanFeedbackResult`.
- `learn=True` stores distilled lessons from feedback in the flow's `memory` and applies them to later outputs before review.

### Asynchronous review (Slack, a ticket queue, a web UI)

The default provider reads from the console. For a reviewer who answers later, give the decorator a provider that raises `HumanFeedbackPending`. The flow saves its state and pending context, and `kickoff()` returns the pending marker instead of blocking:

```python
from pydantic import BaseModel
from crewai import LLM
from crewai.flow import (
    Flow, start, listen, persist, human_feedback,
    HumanFeedbackPending, HumanFeedbackProvider, PendingFeedbackContext, HumanFeedbackResult,
)
from crewai.flow.persistence import SQLiteFlowPersistence

store = SQLiteFlowPersistence("reviews.db")


class QueueProvider(HumanFeedbackProvider):
    """Records the request somewhere a person will see it, then pauses the flow."""

    def request_feedback(self, context: PendingFeedbackContext, flow: Flow) -> str:
        print(f"[ticket] review needed for flow {context.flow_id}: {context.message}")
        raise HumanFeedbackPending(context=context)


class DraftState(BaseModel):
    draft: str = ""


@persist(store)
class AsyncReviewFlow(Flow[DraftState]):
    @start()
    @human_feedback(
        message="Approve the draft?",
        emit=["approved", "rejected"],
        llm=LLM(model="openai/%%model=openai-small%%"),
        provider=QueueProvider(),
    )
    def write_draft(self):
        self.state.draft = "Quarterly update draft"
        return self.state.draft

    @listen("approved")
    def publish(self, result: HumanFeedbackResult):
        return f"Published: {result.output}"

    @listen("rejected")
    def discard(self, result: HumanFeedbackResult):
        return f"Discarded: {result.feedback}"


# Request 1: run until the review point. kickoff() returns the pending marker.
outcome = AsyncReviewFlow().kickoff()
assert isinstance(outcome, HumanFeedbackPending)
flow_id = outcome.context.flow_id

# Request 2 (later, maybe another process): restore and resume with the reviewer's answer.
paused = AsyncReviewFlow.from_pending(flow_id, store)
print(paused.resume("Looks good, ship it"))
```

In an async application use `await paused.resume_async(...)`. To change the provider for every flow in a process (for example in a deployment), set `crewai.flow.flow_config.hitl_provider`.

## Conversational flows

A conversational flow handles one chat turn per `handle_turn()` call and keeps the message history in its state. Decorating the class with `ConversationConfig` turns the mode on. A built-in `converse` route answers general messages with the configured LLM; your own `@listen("label")` methods become extra routes, and the same LLM picks the route for each turn using the method docstrings as descriptions.

```python
from crewai import LLM
from crewai.flow import Flow, listen, persist, ConversationConfig, ConversationState
from crewai.flow.persistence import SQLiteFlowPersistence


@ConversationConfig(
    llm=LLM(model="openai/%%model=openai-small%%"),   # answers general chat and picks the route
    system_prompt="You are the support assistant for Acme Hosting. Be brief.",
)
@persist(SQLiteFlowPersistence("chat.db"))
class SupportChat(Flow[ConversationState]):
    @listen("refund")
    def handle_refund(self):
        """The customer wants a refund or to cancel a paid plan."""
        return "I've opened a refund request; billing will reply within one working day."


chat = SupportChat()
print(chat.handle_turn("Do you support IPv6?", session_id="customer-42"))              # converse route
print(chat.handle_turn("I'd like a refund for last month", session_id="customer-42"))  # refund route
print(len(chat.state.messages), chat.state.last_intent)                                # 4 refund
```

- `session_id` becomes `state.id`; with `@persist`, a later `handle_turn` on a new instance with the same id continues the conversation.
- `ConversationState` holds `messages`, `last_user_message`, `last_intent`, `ended` and private `events`/`agent_threads`. A user message routed to the built-in `end` label sets `ended=True`.
- `chat.chat()` runs a terminal REPL around `handle_turn`; `stream_turn()` streams a turn as frames. Pass a `RouterConfig` to `ConversationConfig(router=...)` to change the routing prompt, give explicit route descriptions or set a fallback intent; override `route_turn()` to route without an LLM.
- `ConversationConfig(respect_context_window=True)` is the default: before each built-in chat or router LLM call, a long history is summarised with the model's summariser, and a provider context-limit error gets one compact-and-retry. The transcript in `state.messages` is not changed. Pass `False` to send the full history as it is (compaction was added in 1.15.31).
- Conversational flows emit `ConversationTurnStartedEvent`, `ConversationRouteSelectedEvent`, `ConversationTurnCompletedEvent` and related events for listeners.

## Declarative flows

A flow can also be written as YAML (schema `crewai.flow/v1`) and run without a Python class. `crewai create flow <name> --declarative` scaffolds a project with `src/<name>/flow.yaml` and `[tool.crewai] definition = ...` in `pyproject.toml`.

```yaml
schema: crewai.flow/v1
name: TopicFlow
description: Read a topic, then report its length.

state:
  type: dict
  default:
    topic: AI agents

methods:
  read_topic:
    start: true
    do:
      call: expression
      expr: state.topic
  report:
    listen: read_topic
    do:
      call: expression
      expr: '"topic has " + string(size(outputs.read_topic)) + " characters"'
```

```python
from crewai.flow import Flow

flow = Flow.from_declaration(path="flow.yaml")
print(flow.kickoff(inputs={"topic": "Vector Databases"}))   # topic has 16 characters
```

- From the CLI: `crewai run --definition flow.yaml --inputs '{"topic": "AI"}'`.
- Method actions are `expression` (CEL, evaluated against `state` and `outputs`), `crew` (inline or a JSON crew declaration), `agent`, `tool`, `code` (a Python reference), `script` (trusted inline Python) and `each` (loop over a list). Methods take the same `start`, `listen` and `router` keys as the decorators.
- A Python flow's structure is available as `MyFlow.flow_definition()`; `.to_dict()` shows the generated definition, which is a quick way to check what the decorators produced.

## Running, visualising and streaming

| Call | Use |
|---|---|
| `flow.kickoff(inputs=...)` | Run synchronously; returns the last method's output. |
| `await flow.kickoff_async(inputs=...)` / `await flow.akickoff(...)` | Run inside an event loop. |
| `flow.kickoff(from_checkpoint=CheckpointConfig(restore_from=...))` | Resume a checkpointed run. |
| `Flow(stream=True)` then `kickoff()` | Returns a stream session of frames instead of the final value. |
| `flow.plot("my_flow.html", show=False)` | Writes an interactive HTML graph and returns its path (a temporary directory in %%version=crewai%%). |
| `crewai flow plot` | The same from a flow project. |

Async methods (`async def`) are supported anywhere a sync method is.

## Testing flows

Routing and state handling are plain Python, so most of a flow can be tested without a model: kick it off with inputs that select each branch and assert on the output or `flow.state`. For methods that call crews, use a stub `BaseLLM`; [Production → Testing without a model](/crewai-guide/crewai_production_guide/#testing-without-a-model) has a complete pytest example.

## Gotchas

- **Validation happens at instantiation.** Self-triggering listeners and other wiring errors surface as a `ValidationError` from `MyFlow()`, before anything runs.
- **Don't assign `self.state`.** Mutate its fields or keys.
- **Fields need defaults** on a structured state model: a field without one makes the flow fail with a `ValidationError` ("Field required") even when you pass that key in `inputs`, because the state is built empty before the inputs are applied. Kickoff inputs that match no field are silently ignored, so a misspelt key leaves the default in place.
- **`initial_state` is not the kickoff inputs.** Read inputs from `self.state`.
- **`@listen` takes one argument.** Use `and_` / `or_` for several.
- **A router's return value is a label, not data.** Put anything the next step needs in state.
- **`@persist` needs a stable id.** Resume with `inputs={"id": ...}`; a new instance without it starts a new run.

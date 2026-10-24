---
title: "Strands Agents (Python): Hooks, Interventions and Interrupts"
description: "Extending the Strands agent loop in Python: hook events and what each can change, hook providers, plugins and Agent Skills, intervention handlers for policy, human-in-the-loop interrupts, retries and middleware."
framework: "strands"
language: "python"
sidebar:
  label: "Hooks and interrupts"
  order: 3
---

Checked against `strands-agents` %%version=strands-agents%% (%%versions-checked%%). Every example ran against an offline Bedrock stub.

Strands has four layers for changing what the agent loop does, from the most general to the most specific:

| Layer | Use it for |
|-------|-----------|
| **Hooks** | Typed callbacks on lifecycle events: logging, metrics, rewriting inputs and results, cancelling or retrying calls |
| **Plugins** | A named bundle of hooks and tools you can add to any agent in one line |
| **Interventions** | Policy decisions (allow, deny, ask a person, guide, transform) evaluated in order before tools and model calls |
| **Interrupts** | Pausing the agent until a person answers, then resuming exactly where it stopped |

## Hook events

| Event | Fires | Writable fields |
|-------|-------|-----------------|
| `AgentInitializedEvent` | End of `Agent.__init__` | |
| `BeforeInvocationEvent` | Start of each call | `messages` (the new input), `cancel` |
| `AfterInvocationEvent` | End of each call, success or failure | `resume` (start another call with this input) |
| `MessageAddedEvent` | A message was appended to `agent.messages` | |
| `MessageUpdatedEvent` | The framework replaced a message already in the history (`tracking_id`, `message`) | |
| `BeforeModelCallEvent` | Before each model call | `cancel` |
| `AfterModelCallEvent` | After each model call (`stop_response` or `exception`) | `retry` |
| `BeforeToolsEvent` | Before a batch of tool calls from one turn | `cancel` |
| `BeforeToolCallEvent` | Before each tool call | `tool_use` (change the input), `selected_tool` (swap the tool), `cancel_tool` |
| `AfterToolCallEvent` | After each tool call (`result`, `exception`, `duration`) | `result`, `retry` |
| `AfterToolsEvent` | After the batch | `end_turn` (stop the loop after these tools) |

Every event has `event.agent`. Writing to a field that is not writable raises `AttributeError`. `After*` events run their callbacks in reverse registration order, so the first hook registered is the outermost, like nested `try`/`finally` blocks. Multi-agent orchestrators have their own events (`BeforeNodeCallEvent`, `AfterNodeCallEvent`, `BeforeMultiAgentInvocationEvent`, ...), covered on the [multi-agent page](/strands-agents-guide/python/strands_multi_agent/#hooks-for-orchestrators).

## Registering hooks

A plain function with a typed event parameter is enough. Pass it in `hooks=`, or add it later with `agent.add_hook()`; the event type is taken from the annotation, and a union registers for each type:

```python
import time

from strands import Agent, tool
from strands.hooks import AfterToolCallEvent, BeforeModelCallEvent, BeforeToolCallEvent


@tool
def get_weather(city: str) -> str:
    """Get the weather.

    Args:
        city: City name.
    """
    return f"Rain in {city}"


def log_tool(event: BeforeToolCallEvent) -> None:
    print("tool:", event.tool_use["name"], event.tool_use["input"])


def time_tool(event: AfterToolCallEvent) -> None:
    status = event.result["status"]
    print(f"{event.tool_use['name']} -> {status} in {event.duration:.3f}s")


agent = Agent(
    model="us.anthropic.%%model=anthropic%%",
    tools=[get_weather],
    hooks=[log_tool, time_tool],
    callback_handler=None,
)


def count_model_calls(event: BeforeModelCallEvent) -> None:
    calls = event.agent.state.get("model_calls") or 0
    event.agent.state.set("model_calls", calls + 1)


agent.add_hook(count_model_calls)
agent("What's the weather in Leeds?")
print("model calls:", agent.state.get("model_calls"))
```

`agent.add_hook(callback, EventType)` names the event explicitly, `add_hook(callback, [A, B])` registers for several, and `order=` sets priority (lower runs first; `HookOrder.SDK_FIRST`, `DEFAULT`, `SDK_LAST`). Callbacks may be `async def`.

### Hook providers

Group related callbacks in a class with `register_hooks`:

```python
from strands import Agent
from strands.hooks import AfterInvocationEvent, BeforeInvocationEvent, HookProvider, HookRegistry


class RequestTimer(HookProvider):
    def __init__(self) -> None:
        self.durations: list[float] = []

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeInvocationEvent, self.start)
        registry.add_callback(AfterInvocationEvent, self.stop)

    def start(self, event: BeforeInvocationEvent) -> None:
        import time
        event.invocation_state["started"] = time.perf_counter()

    def stop(self, event: AfterInvocationEvent) -> None:
        import time
        self.durations.append(time.perf_counter() - event.invocation_state["started"])


timer = RequestTimer()
agent = Agent(model="us.anthropic.%%model=anthropic%%", hooks=[timer], callback_handler=None)
agent("Hello")
print(f"{timer.durations[0]:.3f}s")
```

`invocation_state` is shared by every hook and tool in one call, which makes it the place to carry per-request data between a `Before` and an `After` hook.

## Common hook patterns

### Block or rewrite a tool call

```python
from strands import Agent, tool
from strands.hooks import BeforeToolCallEvent


@tool
def run_sql(query: str) -> str:
    """Run a SQL query against the reporting database.

    Args:
        query: SQL to run.
    """
    return "3 rows"


def read_only_sql(event: BeforeToolCallEvent) -> None:
    if event.tool_use["name"] != "run_sql":
        return
    query = event.tool_use["input"]["query"]
    if not query.lstrip().lower().startswith("select"):
        event.cancel_tool = "Only SELECT statements are allowed."   # the model receives this as an error result
        return
    event.tool_use["input"]["query"] = query.rstrip(";") + " LIMIT 1000"   # rewrite the input


agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[run_sql], hooks=[read_only_sql], callback_handler=None)
agent("Delete all rows from orders")
```

### Redact a tool result

```python
from strands import Agent, tool
from strands.hooks import AfterToolCallEvent


@tool
def get_customer(customer_id: str) -> dict:
    """Fetch a customer record.

    Args:
        customer_id: Customer ID.
    """
    return {"id": customer_id, "name": "Priya", "card": "4111 1111 1111 1111"}


def redact_cards(event: AfterToolCallEvent) -> None:
    for block in event.result["content"]:
        if "text" in block:
            block["text"] = block["text"].replace("4111 1111 1111 1111", "**** **** **** 1111")


agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[get_customer], hooks=[redact_cards], callback_handler=None)
```

### Inspect or cancel a request

```python
from strands import Agent
from strands.hooks import BeforeInvocationEvent


def refuse_secrets(event: BeforeInvocationEvent) -> None:
    text = " ".join(
        block.get("text", "") for message in (event.messages or []) for block in message["content"]
    )
    if "password" in text.lower():
        event.cancel = "Please don't share passwords here."


agent = Agent(model="us.anthropic.%%model=anthropic%%", hooks=[refuse_secrets], callback_handler=None)
result = agent("My password is hunter2, can you remember it?")
print(result)               # Please don't share passwords here.
print(result.stop_reason)   # end_turn
```

A cancelled invocation never calls the model: the cancel message becomes the assistant's answer, and the user's input is not added to the history. The other `cancel` fields differ slightly: `BeforeModelCallEvent.cancel` skips that model call and ends the turn with the message as the answer (the user's input stays in the history), while `BeforeToolsEvent.cancel` gives every tool call in the batch an error result carrying the message and lets the loop continue, as `cancel_tool` does for a single call.

### Retrying tool and model calls

Setting `retry = True` on `AfterToolCallEvent` runs the tool again; on `AfterModelCallEvent` it discards the response and calls the model again. Keep your own counter so a hook cannot loop for ever:

```python
from strands import Agent, tool
from strands.hooks import AfterToolCallEvent

attempts = {"n": 0}


@tool
def flaky_lookup(key: str) -> str:
    """Look up a key in a flaky service.

    Args:
        key: Key to look up.
    """
    attempts["n"] += 1
    if attempts["n"] < 3:
        raise ConnectionError("service unavailable")
    return f"value for {key}"


def retry_connection_errors(event: AfterToolCallEvent) -> None:
    retries = event.invocation_state.setdefault("tool_retries", 0)
    if isinstance(event.exception, ConnectionError) and retries < 3:
        event.invocation_state["tool_retries"] = retries + 1
        event.retry = True


agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[flaky_lookup], hooks=[retry_connection_errors], callback_handler=None)
agent("Look up 'alpha'")
print("tool ran", attempts["n"], "times")
```

Streamed events from a discarded attempt have already reached `stream_async` consumers; only the final attempt's result is added to the history.

Model throttling is retried for you by the agent's `retry_strategy`, a `ModelRetryStrategy(max_attempts=6, initial_delay=4, max_delay=240)` by default (seconds, exponential backoff). Pass your own, or `retry_strategy=None` to turn retries off.

### Keep going after the answer

`AfterInvocationEvent.resume` starts another call with the given input, which is a compact way to build a self-checking loop:

```python
from strands import Agent
from strands.hooks import AfterInvocationEvent


def review_once(event: AfterInvocationEvent) -> None:
    if not event.invocation_state.get("reviewed"):
        event.invocation_state["reviewed"] = True
        event.resume = "Check your answer for mistakes and give a corrected final version."


agent = Agent(model="us.anthropic.%%model=anthropic%%", hooks=[review_once], callback_handler=None)
print(agent("What is 17 * 23?"))
```

## Plugins

A plugin bundles hooks and tools. Methods marked with `@hook` are registered by event type, and `@tool` methods become tools:

```python
from strands import Agent, tool
from strands.hooks import BeforeToolCallEvent
from strands.plugins import Plugin, hook


class AuditPlugin(Plugin):
    name = "audit"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    @hook
    def record(self, event: BeforeToolCallEvent) -> None:
        self.calls.append(event.tool_use["name"])

    @tool
    def audit_log(self) -> list:
        """Return the tools called so far in this session."""
        return self.calls


audit = AuditPlugin()
agent = Agent(model="us.anthropic.%%model=anthropic%%", plugins=[audit], callback_handler=None)
print(agent.tool_names)   # ['audit_log']
```

`MultiAgentPlugin` is the equivalent for `Swarm` and `Graph`. Strands ships several plugins in `strands.vended_plugins`, described below: `AgentSkills`, `ContextInjector`, `GoalLoop`, `ContextOffloader` and steering handlers.

### Agent Skills

A skill is a folder with a `SKILL.md`: YAML front matter with `name` and `description`, then instructions. `AgentSkills` puts each skill's name and description in the system prompt and gives the model a `skills` tool that loads the full instructions only when needed, so many skills cost few tokens until one is used:

```python
from pathlib import Path

from strands import Agent, AgentSkills

skill = Path("skills/refund-policy")
skill.mkdir(parents=True, exist_ok=True)
(skill / "SKILL.md").write_text(
    "---\n"
    "name: refund-policy\n"
    "description: Rules for approving refunds. Use before issuing any refund.\n"
    "---\n"
    "Refunds over £100 need a manager's approval. Never refund gift cards.\n"
)

agent = Agent(model="us.anthropic.%%model=anthropic%%", plugins=[AgentSkills(skills=["./skills"])], callback_handler=None)
print(agent.tool_names)   # ['skills']
```

Skills can also be `Skill` objects or `https://` URLs of a `SKILL.md`.

### Context injection

`ContextInjector` adds text to the model input for one call without storing it in the history or the session. Use it for values that go stale, such as the current time or the caller's account status, which would be wrong if replayed from a saved conversation. The callback may be sync or async and returns the text (or `None` to skip). The default trigger, `"userTurn"`, injects only when the latest message is a fresh user message; `trigger="everyTurn"` injects before every model call, including mid-task tool turns:

```python
import datetime

from strands import Agent
from strands.vended_plugins.context_injector import ContextInjector

agent = Agent(
    model="us.anthropic.%%model=anthropic%%",
    plugins=[ContextInjector(lambda context: f"<now>{datetime.datetime.now(datetime.timezone.utc).isoformat()}</now>")],
    callback_handler=None,
)
agent("What time is it?")
print(agent.messages[0]["content"])   # only the user's text: the injected block is not kept
```

The injected text reaches the model verbatim, so escape any field a user influences. If the callback raises, injection is skipped and the model call goes ahead. `current_time` in `strands-agents-tools` is deprecated in favour of this.

### Goal loops

`GoalLoop` re-runs the agent until its answer meets a goal. After each invocation it checks the answer; if the check fails, it feeds the feedback back as a new user message and continues, until the goal is met, `max_attempts` is reached or `timeout` (seconds) passes. The goal is either a sentence, judged by a model (`judge=JudgeConfig(model=...)`, otherwise the agent's `aux_model`, otherwise its `model`), or a function that returns `True`, `False` or `{"passed": ..., "feedback": ...}`:

```python
from strands import Agent
from strands.vended_plugins.goal import GoalLoop


def at_most_fifty_words(response, agent):
    text = " ".join(block["text"] for block in response["content"] if "text" in block)
    words = len(text.split())
    if words <= 50:
        return True
    return {"passed": False, "feedback": f"Too long ({words} words). Cut it to 50 words or fewer."}


goal = GoalLoop(goal=at_most_fifty_words, max_attempts=3, timeout=60.0)
agent = Agent(model="us.anthropic.%%model=anthropic%%", plugins=[goal], callback_handler=None)

print(agent("Explain how a rainbow forms."))
print(goal.last_result(agent))   # GoalResult(passed=..., stop_reason='satisfied' | 'max_attempts' | 'timeout', attempts=[...])
```

`max_attempts` and `timeout` both default to infinity, so always set one: a goal the model cannot meet would otherwise loop for ever. By default each retry keeps the earlier attempts in the conversation, which costs tokens; `preserve_context=False` restarts every attempt from the original conversation. `GoalLoop` uses `AfterInvocationEvent.resume`, so attaching a second one to the same agent raises `RuntimeError`. For a one-off retry, the `resume` hook shown earlier is enough.

### Offloading large tool results

A tool that returns a whole document or query result can fill the context window in one step. `ContextOffloader` intercepts results above `max_result_tokens` (2,500 by default), saves the full content to a storage backend, and leaves the model a preview (`preview_tokens`, 1,000 by default) with a reference. With `include_retrieval_tool=True` (the default) the model also gets a `retrieve_offloaded_content` tool that fetches a regex match or a line range instead of the whole thing. `evict_after_cycles` (20 by default; `None` disables it) deletes stored content once it is that many agent-loop cycles old:

```python
from strands import Agent, tool
from strands.vended_plugins.context_offloader import ContextOffloader, InMemoryStorage


@tool
def big_report() -> str:
    """Return the full nightly report."""
    return "line of report data\n" * 4000


agent = Agent(
    model="us.anthropic.%%model=anthropic%%",
    tools=[big_report],
    plugins=[ContextOffloader(storage=InMemoryStorage(), max_result_tokens=500, preview_tokens=100)],
    callback_handler=None,
)
print(agent.tool_names)     # ['big_report', 'retrieve_offloaded_content']
agent("Summarise tonight's report.")
```

Use `FileStorage("./artifacts")` or `S3Storage` (both from the same module) to keep offloaded content beyond the process; `should_offload=` takes a function to offload only some tools.

### Steering

A steering handler gives the agent just-in-time guidance instead of a long system prompt. Subclass `SteeringHandler` and override `steer_before_tool` (to judge a tool call) and/or `steer_after_model` (to judge a model response). Return `Proceed(reason=...)`, `Guide(reason=...)` or, for tool calls only, `Interrupt(reason=...)`. A `Guide` on a tool call cancels it and tells the model why, so it can try something else; on a model response it discards the response and calls the model again with the guidance added. `Interrupt` pauses for a person through the [interrupt](#interrupts) mechanism, and a "no" cancels the call:

```python
from strands import Agent, tool
from strands.vended_plugins.steering import Guide, Proceed, SteeringHandler


@tool
def send_email(to: str, body: str) -> str:
    """Send an email.

    Args:
        to: Recipient.
        body: Message body.
    """
    return f"sent to {to}"


class InternalOnly(SteeringHandler):
    name = "internal-only"

    async def steer_before_tool(self, *, agent, tool_use, **kwargs):
        recipient = tool_use["input"].get("to", "")
        if tool_use["name"] == "send_email" and not recipient.endswith("@example.com"):
            return Guide(reason="External recipients need review. Save a draft and ask the user to confirm.")
        return Proceed(reason="internal recipient")


agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[send_email], plugins=[InternalOnly()], callback_handler=None)
agent("Email pat@other.org that the report is ready.")
```

`LLMSteeringHandler(system_prompt=...)` makes the same decisions with a model reading your rules in plain language; it judges with `model=` if given, otherwise the agent's `aux_model`, otherwise its `model`, and keeps a ledger of earlier tool calls as context by default. A handler that raises is ignored (the action proceeds), so keep checks simple and test them. For policy that must hold, [intervention handlers](#interventions) are the stricter tool: they run in a fixed order, can `Deny`, and can fail closed with `on_error="deny"`.

## Interventions

An intervention handler makes a policy decision at a lifecycle point and returns an action. Handlers run in the order given; a `Deny` stops evaluation immediately, so put cheap checks (authorisation) before expensive ones (an LLM judge).

| Action | Effect (on `before_tool_call`) |
|--------|-------------------------------|
| `Proceed()` | Carry on |
| `Deny(reason=...)` | Cancel the tool call; the model sees `DENIED: <reason>` |
| `Confirm(prompt=...)` | Pause with an interrupt and ask a person; cancel unless they answer yes |
| `Guide(feedback=...)` | Cancel the call and send the feedback to the model so it tries differently |
| `Transform(apply=fn)` | Modify the event (for example the tool input) and carry on |

Override `before_invocation`, `before_model_call`, `after_model_call`, `before_tool_call` or `after_tool_call`; each may be sync or async.

```python
from strands import Agent, InterventionHandler, tool
from strands.interventions import Confirm, Deny, Proceed


@tool
def delete_bucket(bucket: str) -> str:
    """Delete an S3 bucket.

    Args:
        bucket: Bucket name.
    """
    return f"deleted {bucket}"


class Approvals(InterventionHandler):
    name = "approvals"

    def before_tool_call(self, event, **kwargs):
        if not event.tool_use["name"].startswith("delete_"):
            return Proceed()
        if event.invocation_state.get("read_only"):
            return Deny(reason="read-only caller")
        return Confirm(prompt=f"Allow {event.tool_use['name']} with {event.tool_use['input']}?")


agent = Agent(
    model="us.anthropic.%%model=anthropic%%",
    tools=[delete_bucket],
    interventions=[Approvals()],
    callback_handler=None,
)

result = agent("Delete the bucket called scratch-data")
if result.stop_reason == "interrupt":
    interrupt = result.interrupts[0]
    print(interrupt.name, "-", interrupt.reason)
    result = agent([{"interruptResponse": {"interruptId": interrupt.id, "response": "yes"}}])
print(result)
```

`Confirm` accepts `True`, `"y"` or `"yes"` by default (`evaluate=` changes that). A handler that raises blocks execution by default; set `on_error` to `"deny"` (fail closed) or `"proceed"` (fail open) to change that. Ready-made handlers live in `strands.vended_interventions`: `hitl` (human approval with an optional risk classifier) and `cedar` (Cedar policy authorisation, `cedar` extra).

## Interrupts

An interrupt pauses the agent so a person (or another system) can answer. The agent returns with `stop_reason == "interrupt"` and `result.interrupts`, each with an `id`, a `name` and a `reason`. You resume by calling the agent with `interruptResponse` blocks; the paused tool or hook runs again, and this time `interrupt()` returns the response.

### From a tool

```python
from strands import Agent, ToolContext, tool


@tool(context=True)
def refund(order_id: str, amount: float, tool_context: ToolContext) -> str:
    """Refund an order.

    Args:
        order_id: Order to refund.
        amount: Amount in GBP.
    """
    if amount > 100:
        decision = tool_context.interrupt("refund-approval", reason={"order_id": order_id, "amount": amount})
        if decision != "approve":
            return f"Refund of £{amount} for {order_id} was not approved"
    return f"Refunded £{amount} for {order_id}"


agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[refund], callback_handler=None)
result = agent("Refund £250 on ORD-7")

while result.stop_reason == "interrupt":
    responses = []
    for interrupt in result.interrupts:
        answer = input(f"{interrupt.name} {interrupt.reason} approve? [y/n] ")  # or a ticket, UI, queue
        responses.append({"interruptResponse": {"interruptId": interrupt.id,
                                                "response": "approve" if answer == "y" else "deny"}})
    result = agent(responses)

print(result)
```

Code before `interrupt()` runs twice (once before the pause, once on resume), so keep side effects after it. Interrupt names must be unique within one tool call or hook. `interrupt(name, reason, response=...)` with a response supplied up front does not pause at all, which helps in tests.

### From a hook

`BeforeToolCallEvent` and `BeforeToolsEvent` can interrupt too. This approves every call to a sensitive tool, once per tool name per session, and remembers the answer in `agent.state`:

```python
from strands import Agent, tool
from strands.hooks import BeforeToolCallEvent


@tool
def send_email(to: str, body: str) -> str:
    """Send an email.

    Args:
        to: Recipient.
        body: Message body.
    """
    return f"sent to {to}"


def approve_sensitive(event: BeforeToolCallEvent) -> None:
    name = event.tool_use["name"]
    approved = event.agent.state.get("approved_tools") or []
    if name != "send_email" or name in approved:
        return
    answer = event.interrupt(f"approve-{name}", reason=f"Allow {name}?")
    if answer == "always":
        event.agent.state.set("approved_tools", approved + [name])
    elif answer != "once":
        event.cancel_tool = f"{name} was not approved"


agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[send_email], hooks=[approve_sensitive], callback_handler=None)
result = agent("Email ops@example.com that the deploy finished")
print(result.stop_reason, [i.name for i in result.interrupts])
result = agent([{"interruptResponse": {"interruptId": result.interrupts[0].id, "response": "always"}}])
print(result.stop_reason, agent.state.get("approved_tools"))
```

### Across processes

Pending interrupts are part of the agent's saved state. With a session manager, the agent can stop with an interrupt in one request and be resumed by a new agent with the same session ID in another, for example after an approval arrives by email hours later. The [approval workflow recipe](/strands-agents-guide/python/strands_recipes/#approval-workflow-across-requests) shows this end to end.

The vended `handoff_to_user` tool (`strands.vended_tools`) lets the model itself decide to stop and ask the user a question, using the same mechanism.

## Middleware

Middleware wraps a whole stage of the loop rather than observing it: `InvokeModelStage` (a model call) and `ExecuteToolStage` (a tool call). A wrap handler is an async generator that receives the context and the next handler, so it can time, cache, short-circuit or retry the stage. `.Input` and `.Output` sub-stages transform the context or the result:

```python
import time

from strands import Agent
from strands.middleware import InvokeModelStage

timings: list[float] = []


async def time_model_calls(context, next_fn):
    started = time.perf_counter()
    async for event in next_fn(context):
        yield event
    timings.append(time.perf_counter() - started)


def be_concise(context):
    return context.replace(system_prompt="Answer in one sentence.")


agent = Agent(model="us.anthropic.%%model=anthropic%%", callback_handler=None)
agent.add_middleware(InvokeModelStage, time_model_calls)
agent.add_middleware(InvokeModelStage.Input, be_concise)
agent("What is Bedrock?")
print(len(timings), "model call(s)")
```

Prefer hooks for observing and simple edits; reach for middleware when you need to control whether and how the stage itself runs. Middleware cannot be removed once added.

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | 1.62.0 | New page. Hook events with writable fields, hook providers, blocking, redaction, retries and resume patterns, plugins and Agent Skills, intervention handlers, interrupts from tools and hooks, and middleware. |

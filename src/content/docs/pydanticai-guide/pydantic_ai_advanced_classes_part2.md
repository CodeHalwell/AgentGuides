---
title: "PydanticAI: Advanced Class Deep Dives (Part 2)"
description: "Source-verified deep dives into AgentSpec, SkipModelRequest/SkipToolExecution/SkipToolValidation, ProcessEventStream, HandleDeferredToolCalls, PreparedToolset, DeferredLoadingToolset, InstructionPart, IncludeToolReturnSchemas, NativeOutput/PromptedOutput/TextOutput advanced patterns."
framework: pydanticai
language: python
---

# PydanticAI: Advanced Class Deep Dives (Part 2)

Verified against **pydantic-ai==1.102.0** — source modules inspected directly from the installed package.

This guide covers ten classes and patterns that receive light treatment elsewhere. Every example is derived directly from the installed source code.

---

## 1. `AgentSpec` — declarative YAML/JSON agent configuration

Source: `pydantic_ai/agent/spec.py`

`AgentSpec` is a Pydantic `BaseModel` that lets you construct an `Agent` from YAML or JSON rather than Python constructor calls. It is the serialisation format for the spec-based workflow and supports capabilities, instructions, model settings, and retry budgets.

### Why use `AgentSpec`?

- **GitOps / config-as-code** — store agent definitions in YAML files alongside infra config.
- **Dynamic loading** — load different agents at runtime based on environment or feature flags.
- **Schema validation** — the spec ships with a `$schema` reference; IDEs validate YAML as you type.
- **Cross-service sharing** — share agent definitions as JSON payloads over HTTP.

### Minimal YAML workflow

```python
# Save a spec to YAML
from pathlib import Path
from pydantic_ai.agent import AgentSpec

spec = AgentSpec(
    model='openai:gpt-4o',
    name='SupportBot',
    instructions='You are a concise technical support assistant.',
    retries=2,
)
spec.to_file('support_agent.yaml')

# Load it back and build an Agent
loaded = AgentSpec.from_file('support_agent.yaml')
agent = loaded.to_agent()
result = agent.run_sync('How do I reset my password?')
print(result.output)
```

The generated YAML looks like:
```yaml
# yaml-language-server: $schema=./support_agent_schema.json
model: openai:gpt-4o
name: SupportBot
instructions: You are a concise technical support assistant.
retries: 2
```

### Full field reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str \| None` | `None` | Model string, e.g. `'openai:gpt-4o'` |
| `name` | `str \| None` | `None` | Agent display name |
| `description` | `str \| TemplateStr \| None` | `None` | Agent description (supports template variables) |
| `instructions` | `str \| list[str] \| None` | `None` | System-level instruction(s) |
| `retries` | `int \| AgentRetries \| None` | `None` | Retry budget (int = total, or per-type dict) |
| `model_settings` | `dict[str, Any] \| None` | `None` | Forwarded as `ModelSettings` |
| `end_strategy` | `EndStrategy` | `'early'` | When to stop: `'early'` or `'exhaustive'` |
| `tool_timeout` | `float \| None` | `None` | Tool execution timeout in seconds |
| `capabilities` | `list[CapabilitySpec]` | `[]` | Declarative capabilities |
| `metadata` | `dict[str, Any] \| None` | `None` | Arbitrary metadata |
| `deps_schema` | `dict[str, Any] \| None` | `None` | JSON schema for deps (documentation only) |
| `output_schema` | `dict[str, Any] \| None` | `None` | JSON schema for output (documentation only) |

### Loading from a string

```python
from pydantic_ai.agent import AgentSpec

yaml_text = """
model: anthropic:claude-sonnet-4-6
name: Researcher
instructions:
  - You are a concise research assistant.
  - Always cite your sources.
retries: 3
model_settings:
  temperature: 0.2
  max_tokens: 1000
"""

spec = AgentSpec.from_text(yaml_text, fmt='yaml')
agent = spec.to_agent()
print(f'Agent name: {agent.name}')
```

### Loading from JSON

```python
import json
from pydantic_ai.agent import AgentSpec

payload = {
    "model": "openai:gpt-4o",
    "instructions": "You are a code reviewer. Be constructive and specific.",
    "retries": 2,
    "model_settings": {"temperature": 0.0}
}

spec = AgentSpec.from_dict(payload)
agent = spec.to_agent()
result = agent.run_sync('Review: x = 1+1')
print(result.output)
```

### `to_file` with schema generation

When you call `to_file()`, PydanticAI automatically saves a companion `_schema.json` alongside the YAML so editors with the `yaml-language-server` extension validate field names and values:

```python
from pathlib import Path
from pydantic_ai.agent import AgentSpec

spec = AgentSpec(
    model='openai:gpt-4o',
    instructions='Be helpful and concise.',
    model_settings={'temperature': 0.3, 'max_tokens': 500},
    retries=3,
)

# Saves agent.yaml + agent_schema.json in the current directory
spec.to_file('agent.yaml')

# Control schema location
spec.to_file(
    'config/agent.yaml',
    schema_path='config/agent_schema.json',  # relative or absolute path
)

# No schema (e.g. already committed)
spec.to_file('agent.yaml', schema_path=None)
```

### Capabilities in YAML

The `capabilities` list supports any spec-serialisable capability. Built-in serialisable capabilities include `Thinking`, `ReinjectSystemPrompt`, `ProcessHistory`, `IncludeToolReturnSchemas`, and `Instrumentation`:

```yaml
# agent.yaml
model: anthropic:claude-opus-4-7
instructions: Reason carefully about each question.
capabilities:
  - Thinking: medium        # short form: single positional arg
  - ReinjectSystemPrompt    # no args — bare name
  - IncludeToolReturnSchemas
  - Instrumentation:        # keyword args via dict
      event_mode: body
```

```python
from pydantic_ai.agent import AgentSpec

spec = AgentSpec.from_file('agent.yaml')
agent = spec.to_agent()
result = agent.run_sync('Solve: if x+y=10 and x*y=21, find x and y.')
print(result.output)
```

### `AgentRetries` — per-side retry budgets

Pass `AgentRetries` for independent tool and output retry limits:

```python
from pydantic_ai.agent import AgentSpec
from pydantic_ai.agent.abstract import AgentRetries

spec = AgentSpec(
    model='openai:gpt-4o',
    instructions='Be a diligent assistant.',
    retries=AgentRetries(tools=5, output=3),
)
agent = spec.to_agent()
```

In YAML, the equivalent is:
```yaml
retries:
  tools: 5
  output: 3
```

### Dynamic environment-based loading

```python
import os
from pathlib import Path
from pydantic_ai.agent import AgentSpec

def load_agent_for_env():
    """Load different agent configs per deployment environment."""
    env = os.getenv('APP_ENV', 'development')
    spec_path = Path(f'agents/{env}_agent.yaml')

    if not spec_path.exists():
        spec_path = Path('agents/default_agent.yaml')

    spec = AgentSpec.from_file(spec_path)
    return spec.to_agent()

agent = load_agent_for_env()
```

---

## 2. `SkipModelRequest`, `SkipToolExecution`, `SkipToolValidation`

Source: `pydantic_ai/exceptions.py`

These three exception classes let hooks **short-circuit** the normal execution pipeline. Raise them inside `before_*` or `wrap_*` hooks to inject a synthetic result without executing the real operation.

### `SkipModelRequest` — inject a cached model response

Raise `SkipModelRequest(response)` inside `before_model_request` or `wrap_model_request` to bypass the model call and use a pre-built `ModelResponse` instead. Useful for:

- **Response caching** — return a cached response when the prompt is identical.
- **Test injection** — inject deterministic responses without an API key.
- **Circuit breakers** — return a fallback response when a backend is unhealthy.

```python
import asyncio
import hashlib
import json
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import SkipModelRequest
from pydantic_ai.messages import ModelResponse, TextPart, RequestUsage

# Simple in-memory cache
_cache: dict[str, ModelResponse] = {}

def _cache_key(messages) -> str:
    # Use model_dump for stable, field-aware serialisation instead of str()
    content = json.dumps([m.model_dump(mode='json') for m in messages], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()

hooks = Hooks()

@hooks.on.before_model_request
async def return_if_cached(ctx, request_context):
    key = _cache_key(request_context.messages)
    if key in _cache:
        print('[cache HIT]')
        raise SkipModelRequest(_cache[key])  # ← bypass the model
    print('[cache MISS]')
    return request_context  # continue normally

@hooks.on.after_model_request
async def store_in_cache(ctx, *, request_context, response):
    key = _cache_key(request_context.messages)
    _cache[key] = response
    return response

agent = Agent('openai:gpt-4o', capabilities=[hooks])

async def main():
    r1 = await agent.run('What is 1 + 1?')   # cache MISS → API call
    r2 = await agent.run('What is 1 + 1?')   # cache HIT  → no API call
    print(r1.output, r2.output)              # identical outputs

asyncio.run(main())
```

### Building a synthetic `ModelResponse` for `SkipModelRequest`

```python
from pydantic_ai.messages import ModelResponse, TextPart, RequestUsage
import datetime

def make_response(text: str) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content=text)],
        model_name='cache',
        usage=RequestUsage(request_tokens=0, response_tokens=0),
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )

hooks = Hooks()

@hooks.on.before_model_request
async def always_cached(ctx, request_context):
    raise SkipModelRequest(make_response('The answer is 42.'))

agent = Agent('openai:gpt-4o', capabilities=[hooks])
result = agent.run_sync('What is the answer?')
print(result.output)   # The answer is 42.
```

### `SkipToolValidation` — bypass argument parsing

Raise `SkipToolValidation(validated_args)` inside `before_tool_validate` or `wrap_tool_validate` to skip Pydantic argument validation and inject pre-validated args directly. Useful for:

- Providing hard-coded default args when the model produces bad JSON.
- Normalising args before validation (e.g. lowercase field names).

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import SkipToolValidation

hooks = Hooks()

@hooks.on.before_tool_validate(tools=['search'])
async def normalise_args(ctx, *, call, tool_def, raw_args):
    """Ensure 'query' is always lowercase before validation."""
    if isinstance(raw_args, dict) and 'query' in raw_args:
        normalised = {**raw_args, 'query': raw_args['query'].lower()}
        raise SkipToolValidation(normalised)   # ← skip validation, use normalised args
    return raw_args

agent = Agent('openai:gpt-4o', capabilities=[hooks])

@agent.tool_plain
def search(query: str) -> list[str]:
    """Search for items matching the query."""
    return [f'Result for: {query}']

async def main():
    result = await agent.run('Search for PYTHON TUTORIALS')
    print(result.output)   # query will be lowercased

asyncio.run(main())
```

### `SkipToolExecution` — mock tool results

Raise `SkipToolExecution(result)` inside `before_tool_execute` or `wrap_tool_execute` to skip the actual tool function and inject a pre-computed result. Useful for:

- **Testing** — avoid real I/O in unit tests.
- **Dry-run mode** — log what *would* happen without side effects.
- **Sandboxing** — prevent certain tools from executing in restricted contexts.

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import SkipToolExecution

DRY_RUN = True   # flip to False for real execution

hooks = Hooks()

@hooks.on.before_tool_execute(tools=['send_email', 'delete_record'])
async def dry_run_guard(ctx, *, call, tool_def, args):
    if DRY_RUN:
        print(f'[DRY RUN] {tool_def.name}({args}) — skipped')
        raise SkipToolExecution(f'[DRY RUN] {tool_def.name} would have run with {args}')
    return args

agent = Agent('openai:gpt-4o', capabilities=[hooks])

@agent.tool_plain
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f'Email sent to {to}'

@agent.tool_plain
def delete_record(record_id: str) -> bool:
    """Delete a record from the database."""
    return True

async def main():
    result = await agent.run('Send a welcome email to alice@example.com and delete record 42.')
    print(result.output)

asyncio.run(main())
```

### Combining all three skip exceptions — testing harness

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import SkipModelRequest, SkipToolExecution
from pydantic_ai.messages import ModelResponse, TextPart, RequestUsage
import datetime

def scripted_response(text: str) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content=text)],
        model_name='test',
        usage=RequestUsage(request_tokens=10, response_tokens=5),
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )

class ScriptedTestHooks:
    """Hooks that provide fully scripted responses for deterministic tests."""

    def __init__(self, model_responses: list[str], tool_results: dict[str, object]):
        self._model_responses = iter(model_responses)
        self._tool_results = tool_results
        self.hooks = Hooks()
        self.tool_calls_made: list[str] = []

        @self.hooks.on.before_model_request
        async def inject_model(ctx, request_context):
            try:
                text = next(self._model_responses)
                raise SkipModelRequest(scripted_response(text))
            except StopIteration:
                return request_context   # fall through to real model

        @self.hooks.on.before_tool_execute
        async def inject_tool(ctx, *, call, tool_def, args):
            self.tool_calls_made.append(tool_def.name)
            if tool_def.name in self._tool_results:
                raise SkipToolExecution(self._tool_results[tool_def.name])
            return args

# Usage in tests
async def run_test():
    harness = ScriptedTestHooks(
        model_responses=["I'll search for that now.", "Found it! Here's the result."],
        tool_results={'search': ['Result A', 'Result B']},
    )

    agent = Agent('openai:gpt-4o', capabilities=[harness.hooks])

    @agent.tool_plain
    def search(query: str) -> list[str]:
        return ['real result']  # never reached in tests

    result = await agent.run('Find Python tutorials.')
    print(result.output)
    print('Tools called:', harness.tool_calls_made)

asyncio.run(run_test())
```

---

## 3. `ProcessEventStream` — real-time event monitoring

Source: `pydantic_ai/capabilities/process_event_stream.py`

`ProcessEventStream` intercepts the agent's internal event stream — model tokens arriving, tool calls starting, tool results returning. It supports two modes:

- **Observer** (`async def handler(ctx, stream) -> None`): receives events while passing them through unchanged.
- **Processor** (`async def handler(ctx, stream)` returning an `AsyncIterator`): yields events, allowing filtering, transformation, or injection.

When this capability is registered, streaming is automatically enabled — you don't need `run_stream()`.

### Observer mode — log events as they arrive

```python
import asyncio
from collections.abc import AsyncIterable
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import (
    AgentStreamEvent, PartStartEvent, PartDeltaEvent,
    FunctionToolCallEvent, FunctionToolResultEvent, TextPartDelta
)

async def log_events(ctx, stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        if isinstance(event, PartStartEvent):
            print(f'[start] part_index={event.index}')
        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            print(event.delta.content_delta, end='', flush=True)
        elif isinstance(event, FunctionToolCallEvent):
            print(f'\n[tool call] {event.part.tool_name}({event.part.args_as_dict()})')
        elif isinstance(event, FunctionToolResultEvent):
            print(f'[tool result] {str(event.result.content)[:80]}')

agent = Agent(
    'openai:gpt-4o',
    capabilities=[ProcessEventStream(handler=log_events)],
)

@agent.tool_plain
def get_weather(city: str) -> str:
    return f'Sunny, 22°C in {city}'

async def main():
    result = await agent.run('What is the weather in Paris?')
    print('\nFinal output:', result.output)

asyncio.run(main())
```

### Processor mode — transform or filter events

```python
import asyncio
from collections.abc import AsyncIterator
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import (
    AgentStreamEvent, PartDeltaEvent, TextPartDelta
)

async def censor_text(ctx, stream: AsyncIterator[AgentStreamEvent]) -> AsyncIterator[AgentStreamEvent]:
    """Replace any mention of 'secret' in streamed text with '***'."""
    async for event in stream:
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            censored = event.delta.content_delta.replace('secret', '***')
            yield PartDeltaEvent(
                index=event.index,
                delta=TextPartDelta(content_delta=censored),
            )
        else:
            yield event

agent = Agent(
    'openai:gpt-4o',
    capabilities=[ProcessEventStream(handler=censor_text)],
    instructions='Tell me a story involving secrets.',
)

async def main():
    result = await agent.run('Tell me about the secret project.')
    print(result.output)   # 'secret' replaced with '***'

asyncio.run(main())
```

### Building a live progress UI

```python
import asyncio
import sys
from collections.abc import AsyncIterable
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import (
    AgentStreamEvent, PartDeltaEvent, TextPartDelta,
    FunctionToolCallEvent, FunctionToolResultEvent
)

class ProgressDisplay:
    def __init__(self):
        self.tool_call_count = 0
        self.char_count = 0
        self.hooks = ProcessEventStream(handler=self._handler)

    async def _handler(self, ctx, stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            if isinstance(event, FunctionToolCallEvent):
                self.tool_call_count += 1
                print(f'\r⚙ Running tool: {event.part.tool_name}...', end='', flush=True)
            elif isinstance(event, FunctionToolResultEvent):
                print(f'\r✓ Tool complete.                ', flush=True)
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                self.char_count += len(event.delta.content_delta)
                print(f'\r📝 {self.char_count} chars...', end='', flush=True)

display = ProgressDisplay()
agent = Agent('openai:gpt-4o', capabilities=[display.hooks])

@agent.tool_plain
def search(query: str) -> str:
    return f'Top results for: {query}'

async def main():
    result = await agent.run('Search for Python best practices and summarise them.')
    print(f'\nDone! Tools: {display.tool_call_count}, Output chars: {display.char_count}')
    print(result.output)

asyncio.run(main())
```

### Forwarding events to a WebSocket

```python
import asyncio
import json
from collections.abc import AsyncIterable
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import (
    AgentStreamEvent, PartDeltaEvent, TextPartDelta, FunctionToolCallEvent
)

# Simulate a websocket send function
async def ws_send(data: dict) -> None:
    print(f'→ ws: {json.dumps(data)}')

async def forward_to_ws(ctx, stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            await ws_send({'type': 'text_delta', 'delta': event.delta.content_delta})
        elif isinstance(event, FunctionToolCallEvent):
            await ws_send({'type': 'tool_call', 'tool': event.part.tool_name})

agent = Agent(
    'openai:gpt-4o',
    capabilities=[ProcessEventStream(handler=forward_to_ws)],
)

async def main():
    result = await agent.run('Explain async programming briefly.')
    print('Done:', result.output[:100])

asyncio.run(main())
```

---

## 4. `HandleDeferredToolCalls` — auto-resolve deferred tools

Source: `pydantic_ai/capabilities/deferred_tool_handler.py`

`HandleDeferredToolCalls` intercepts `DeferredToolRequests` during an agent run and resolves them inline using a handler function. Without this capability, deferred tool calls bubble up as `DeferredToolRequests` output and require your code to drive a manual loop.

### Auto-approve all deferred calls

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied

async def approve_all(
    ctx: RunContext[None],
    requests: DeferredToolRequests,
) -> DeferredToolResults:
    """Approve every pending tool call automatically."""
    return requests.build_results(approve_all=True)

agent = Agent(
    'openai:gpt-4o',
    capabilities=[HandleDeferredToolCalls(handler=approve_all)],
)

@agent.tool(requires_approval=True)
def send_email(ctx: RunContext[None], to: str, subject: str) -> str:
    """Send an email — requires approval."""
    return f'Email sent to {to}: {subject}'

async def main():
    # No manual approval loop needed — the capability handles it
    result = await agent.run('Send a welcome email to new@example.com')
    print(result.output)

asyncio.run(main())
```

### Selective approval — approve some, deny others

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied

ALLOWED_TOOLS = {'read_file', 'search_docs'}
DENIED_TOOLS = {'delete_file', 'overwrite_file'}

async def selective_approval(
    ctx: RunContext[None],
    requests: DeferredToolRequests,
) -> DeferredToolResults:
    results = {}
    for call in requests.calls:
        if call.tool_name in ALLOWED_TOOLS:
            results[call.tool_call_id] = ToolApproved()
            print(f'✓ Approved: {call.tool_name}')
        elif call.tool_name in DENIED_TOOLS:
            results[call.tool_call_id] = ToolDenied(
                message=f'{call.tool_name} is not permitted in this context.'
            )
            print(f'✗ Denied: {call.tool_name}')
        else:
            # Unknown tool — deny with explanation
            results[call.tool_call_id] = ToolDenied(message=f'{call.tool_name} is unknown.')
    return DeferredToolResults(calls=results)

agent = Agent(
    'openai:gpt-4o',
    capabilities=[HandleDeferredToolCalls(handler=selective_approval)],
)
```

### Human-in-the-loop via console

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied

async def console_approval(
    ctx: RunContext[None],
    requests: DeferredToolRequests,
) -> DeferredToolResults:
    """Ask a human on the console to approve each deferred call."""
    results = {}
    for call in requests.calls:
        print(f'\nTool: {call.tool_name}')
        print(f'Args: {call.args_as_dict()}')
        answer = input('Approve? [y/N] ').strip().lower()
        if answer == 'y':
            results[call.tool_call_id] = ToolApproved()
        else:
            reason = input('Reason for denial (optional): ').strip()
            results[call.tool_call_id] = ToolDenied(message=reason or 'Denied by operator.')
    return DeferredToolResults(calls=results)

agent = Agent(
    'openai:gpt-4o',
    capabilities=[HandleDeferredToolCalls(handler=console_approval)],
)
```

### Async handler that calls an external approval service

```python
import asyncio
import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied

async def remote_approval(
    ctx: RunContext[None],
    requests: DeferredToolRequests,
) -> DeferredToolResults:
    """Submit tool calls to an external approval service and wait for responses."""
    async with httpx.AsyncClient() as client:
        payload = [
            {'id': call.tool_call_id, 'tool': call.tool_name, 'args': call.args_as_dict()}
            for call in requests.calls
        ]
        response = await client.post(
            'https://approvals.example.com/batch',
            json={'calls': payload, 'run_id': ctx.run_id},
            timeout=30.0,
        )
        decisions = response.json()  # [{'id': '...', 'approved': True/False, 'reason': '...'}]

    results = {}
    for decision in decisions:
        if decision['approved']:
            results[decision['id']] = ToolApproved()
        else:
            results[decision['id']] = ToolDenied(message=decision.get('reason', 'Denied'))
    return DeferredToolResults(calls=results)

agent = Agent(
    'openai:gpt-4o',
    capabilities=[HandleDeferredToolCalls(handler=remote_approval)],
)
```

### Handler returning `None` — decline to resolve

If the handler returns `None`, it signals that it doesn't handle these requests. The next capability in the chain gets a chance; if none handle it, the `DeferredToolRequests` bubbles up as output:

```python
async def maybe_handle(ctx, requests: DeferredToolRequests) -> DeferredToolResults | None:
    """Only handle 'safe' tools; let others bubble up."""
    safe_calls = {
        call.tool_call_id: ToolApproved()
        for call in requests.calls
        if call.tool_name in SAFE_TOOLS
    }
    if len(safe_calls) == len(requests.calls):
        return DeferredToolResults(calls=safe_calls)
    return None   # some calls are not safe — let caller decide
```

---

## 5. `PreparedToolset` — dynamic tool definition mutation

Source: `pydantic_ai/toolsets/prepared.py`

`PreparedToolset` wraps any toolset and runs a `(RunContext, list[ToolDefinition]) -> list[ToolDefinition]` function before each model request. Use it to redescribe, add schema annotations, or hide individual tools based on runtime state — without reimplementing the underlying tool functions.

**Important constraint**: the prepare function can modify (descriptions, schema annotations, `strict` mode) or remove tools, but **cannot rename, add, or substitute tools**. Use `RenamedToolset` for renaming and `FunctionToolset.add_function()` for additions.

### Locale-aware tool descriptions

```python
import asyncio
import dataclasses
from dataclasses import dataclass
from pydantic_ai import Agent, FunctionToolset, PreparedToolset, RunContext
from pydantic_ai.tools import ToolDefinition

DESCRIPTIONS = {
    'en': {'search': 'Search the database for documents.', 'get_record': 'Retrieve a single record.'},
    'fr': {'search': 'Rechercher des documents dans la base.', 'get_record': 'Récupérer un enregistrement.'},
    'de': {'search': 'Dokumente in der Datenbank suchen.', 'get_record': 'Einen Datensatz abrufen.'},
}

@dataclass
class UserDeps:
    locale: str = 'en'

tools = FunctionToolset[UserDeps]()

@tools.tool_plain
def search(query: str) -> list[str]:
    """Search the database for documents."""
    return [f'Result: {query}']

@tools.tool_plain
def get_record(record_id: str) -> dict:
    """Retrieve a single record."""
    return {'id': record_id, 'data': 'example'}

async def localise_tools(ctx: RunContext[UserDeps], defs: list[ToolDefinition]) -> list[ToolDefinition]:
    locale_map = DESCRIPTIONS.get(ctx.deps.locale, DESCRIPTIONS['en'])
    return [
        dataclasses.replace(d, description=locale_map.get(d.name, d.description))
        for d in defs
    ]

agent = Agent(
    'openai:gpt-4o',
    deps_type=UserDeps,
    toolsets=[PreparedToolset(tools, prepare_func=localise_tools)],
)

async def main():
    result = await agent.run('Cherche des documents Python.', deps=UserDeps(locale='fr'))
    print(result.output)

asyncio.run(main())
```

### Toggling `strict` mode per environment

```python
import os
from pydantic_ai import Agent, FunctionToolset, PreparedToolset, RunContext
from pydantic_ai.tools import ToolDefinition
from dataclasses import replace

tools = FunctionToolset[None]()

@tools.tool_plain
def calculate(expression: str) -> float:
    """Evaluate a simple arithmetic expression."""
    return float(eval(expression, {'__builtins__': {}}))  # noqa: S307

STRICT = os.getenv('OPENAI_STRICT', 'false') == 'true'

def maybe_strict(_ctx: RunContext[None], defs: list[ToolDefinition]) -> list[ToolDefinition]:
    if STRICT:
        return [replace(d, strict=True) for d in defs]
    return defs

agent = Agent(
    'openai:gpt-4o',
    toolsets=[PreparedToolset(tools, prepare_func=maybe_strict)],
)

result = agent.run_sync('What is 1234 * 5678?')
print(result.output)
```

### Hiding tools based on feature flags

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, FunctionToolset, PreparedToolset, RunContext
from pydantic_ai.tools import ToolDefinition

FEATURE_FLAGS = {'beta_analytics': False, 'v2_search': True}

@dataclass
class Deps:
    beta_user: bool = False

tools = FunctionToolset[Deps]()

@tools.tool_plain
def legacy_search(query: str) -> list[str]:
    """Search using the legacy engine."""
    return [f'legacy: {query}']

@tools.tool(metadata={'feature_flag': 'v2_search'})
def v2_search(ctx: RunContext[Deps], query: str) -> list[str]:
    """Search using the v2 engine (feature-flagged)."""
    return [f'v2: {query}']

@tools.tool(metadata={'feature_flag': 'beta_analytics'})
def beta_analytics(ctx: RunContext[Deps], report: str) -> dict:
    """Generate analytics report (beta users only)."""
    return {'report': report, 'status': 'beta'}

async def feature_filter(ctx: RunContext[Deps], defs: list[ToolDefinition]) -> list[ToolDefinition]:
    visible = []
    for d in defs:
        flag = d.metadata and d.metadata.get('feature_flag') if d.metadata else None
        if flag is None or FEATURE_FLAGS.get(flag, False):
            visible.append(d)
    return visible

agent = Agent(
    'openai:gpt-4o',
    deps_type=Deps,
    toolsets=[PreparedToolset(tools, prepare_func=feature_filter)],
)

async def main():
    # v2_search visible (flag=True), beta_analytics hidden (flag=False)
    result = await agent.run('Search for Python docs.', deps=Deps())
    print(result.output)

asyncio.run(main())
```

### Adding schema annotations for better structured output

```python
import dataclasses
from pydantic_ai import Agent, FunctionToolset, PreparedToolset, RunContext
from pydantic_ai.tools import ToolDefinition

tools = FunctionToolset[None]()

@tools.tool_plain
def lookup_customer(customer_id: int) -> dict:
    """Retrieve customer record."""
    return {'id': customer_id, 'name': 'Alice', 'tier': 'premium'}

def add_return_hints(_ctx: RunContext[None], defs: list[ToolDefinition]) -> list[ToolDefinition]:
    """Inject return schema as a description annotation."""
    annotated = []
    for d in defs:
        enriched_desc = d.description or ''
        if d.name == 'lookup_customer':
            enriched_desc += ' Returns: {id: int, name: str, tier: str}.'
        annotated.append(dataclasses.replace(d, description=enriched_desc))
    return annotated

agent = Agent('openai:gpt-4o', toolsets=[PreparedToolset(tools, prepare_func=add_return_hints)])
result = agent.run_sync('What tier is customer 42?')
print(result.output)
```

---

## 6. `DeferredLoadingToolset` — hide tools until searched

Source: `pydantic_ai/toolsets/deferred_loading.py`

`DeferredLoadingToolset` marks tools with `defer_loading=True`, hiding them from the model's initial context. When the model encounters a task it can't solve with visible tools, it calls the built-in tool search capability to discover deferred tools dynamically. This reduces the upfront token cost of large tool libraries.

### Basic usage — hide all tools

```python
import asyncio
from pydantic_ai import Agent, FunctionToolset, DeferredLoadingToolset

# A large library of 50+ tools
analysis_tools = FunctionToolset[None]()

@analysis_tools.tool_plain
def run_sql(query: str) -> list[dict]:
    """Execute a SQL query against the analytics database."""
    return [{'result': query}]

@analysis_tools.tool_plain
def export_csv(data: list[dict], filename: str) -> str:
    """Export data to a CSV file."""
    return f'Exported {len(data)} rows to {filename}'

@analysis_tools.tool_plain
def send_report(email: str, subject: str, content: str) -> bool:
    """Email a report to a recipient."""
    return True

# Hide ALL tools — model discovers them via tool search
hidden_tools = DeferredLoadingToolset(analysis_tools)
agent = Agent('openai:gpt-4o', toolsets=[hidden_tools])

async def main():
    # Model won't see run_sql, export_csv, send_report upfront
    # It must search for them if needed
    result = await agent.run('What SQL tools are available?')
    print(result.output)

asyncio.run(main())
```

### Selective deferral — hide only expensive tools

```python
from pydantic_ai import Agent, FunctionToolset, DeferredLoadingToolset, RunContext

tools = FunctionToolset[None]()

@tools.tool_plain
def quick_lookup(key: str) -> str:
    """Fast O(1) lookup from cache."""
    return f'cached: {key}'

@tools.tool_plain
def full_text_search(query: str) -> list[str]:
    """Expensive full-text search across 10M documents."""
    return [f'doc about {query}']

@tools.tool_plain
def run_ml_inference(model_name: str, input_data: dict) -> dict:
    """Run inference on a large ML model (slow + expensive)."""
    return {'prediction': 'result'}

# Only defer the two expensive tools; quick_lookup stays visible
deferred_tools = DeferredLoadingToolset(
    tools,
    tool_names=frozenset({'full_text_search', 'run_ml_inference'}),
)

agent = Agent('openai:gpt-4o', toolsets=[deferred_tools])
result = agent.run_sync('Do a quick lookup for "config".')
print(result.output)   # Uses quick_lookup immediately (not deferred)
```

### `FunctionToolset` shortcut: `defer_loading=True`

`FunctionToolset` has a `defer_loading` constructor argument that wraps the toolset in `DeferredLoadingToolset` automatically:

```python
from pydantic_ai import Agent, FunctionToolset

# Equivalent to DeferredLoadingToolset(FunctionToolset(...))
tools = FunctionToolset[None](defer_loading=True)

@tools.tool_plain
def heavy_computation(n: int) -> int:
    """Run an expensive computation."""
    return sum(range(n))

agent = Agent('openai:gpt-4o', toolsets=[tools])
```

---

## 7. `InstructionPart` — cacheable instruction composition

Source: `pydantic_ai/messages.py`

`InstructionPart` is the internal dataclass that represents a single block of instruction text, tagged as *static* or *dynamic*. Static instructions come from literal strings; dynamic ones come from `@agent.instructions` functions, `TemplateStr`, or toolset `get_instructions()` methods.

Provider-level caching (e.g. Anthropic's prompt caching) uses this distinction: static instructions are always cached; dynamic ones are not, since they may change per run.

### When does this matter?

You encounter `InstructionPart` when:
- **Inspecting message history**: the messages returned by `result.all_messages()` contain `InstructionPart` objects inside `ModelRequest`.
- **Building custom message processors**: a `ProcessHistory` capability might inspect instruction parts.
- **Custom `AbstractCapability`**: manipulate instruction ordering or caching hints.

### Composing instructions with caching awareness

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import InstructionPart
from pydantic_ai.capabilities import ProcessHistory

# Static instruction — always cached (fast, cheap)
STATIC_INSTRUCTIONS = "You are a concise technical assistant specialising in Python."

# Dynamic instruction factory — evaluated per run
async def dynamic_instructions(ctx: RunContext) -> str:
    # Fetch user-specific context
    user_tier = getattr(ctx.deps, 'tier', 'free')
    return f"User tier: {user_tier}. Respond at the {user_tier} support level."

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    instructions=STATIC_INSTRUCTIONS,  # stored as static InstructionPart
)

@agent.instructions
async def per_run_instructions(ctx: RunContext) -> str:
    """Dynamic part — not cached."""
    return await dynamic_instructions(ctx)

async def main():
    result = await agent.run('How do I use async/await in Python?')
    # Inspect what ended up as instructions
    for msg in result.all_messages():
        for part in getattr(msg, 'parts', []):
            if isinstance(part, InstructionPart):
                print(f'  [{\"dynamic\" if part.dynamic else \"static\"}] {part.content[:80]}')

asyncio.run(main())
```

### Sorting for optimal caching

`InstructionPart.sorted()` places static parts before dynamic ones, which maximises Anthropic's cache hit rate (the model's cache key is a prefix of the message list):

```python
from pydantic_ai.messages import InstructionPart

parts = [
    InstructionPart('You are a helpful assistant.', dynamic=False),     # static
    InstructionPart('Current time: 14:30', dynamic=True),               # dynamic
    InstructionPart('Reply in English only.', dynamic=False),           # static
]

optimised = InstructionPart.sorted(parts)
for p in optimised:
    print(f'  [{"dynamic" if p.dynamic else "static "}] {p.content}')
# [static ] You are a helpful assistant.
# [static ] Reply in English only.
# [dynamic] Current time: 14:30
```

### Joining parts into a single string

`InstructionPart.join()` concatenates parts with a double newline:

```python
from pydantic_ai.messages import InstructionPart

parts = [
    InstructionPart('You are a JSON extraction agent.'),
    InstructionPart('Extract all dates in ISO 8601 format.'),
    InstructionPart('Output as a JSON array.'),
]
combined = InstructionPart.join(parts)
print(combined)
# You are a JSON extraction agent.
#
# Extract all dates in ISO 8601 format.
#
# Output as a JSON array.
```

---

## 8. `IncludeToolReturnSchemas` and `IncludeReturnSchemasToolset`

Source: `pydantic_ai/capabilities/include_return_schemas.py` and `pydantic_ai/toolsets/include_return_schemas.py`

These two classes instruct PydanticAI to include a tool's return type schema in the tool definition sent to the model. For providers that natively support return schemas (Google Gemini), this is a structured field. For others, it is injected as JSON text in the tool's description.

**Use them when you want the model to know what shape a tool returns**, which helps it compose multi-step reasoning more accurately.

### Capability form — `IncludeToolReturnSchemas`

```python
from pydantic import BaseModel
from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.capabilities import IncludeToolReturnSchemas

class WeatherReport(BaseModel):
    city: str
    temp_celsius: float
    condition: str
    humidity_pct: int

tools = FunctionToolset[None]()

@tools.tool_plain
def get_weather(city: str) -> WeatherReport:
    """Fetch weather for a city."""
    return WeatherReport(city=city, temp_celsius=22.0, condition='Sunny', humidity_pct=55)

# Include return schema for ALL tools (default)
agent_all = Agent(
    'openai:gpt-4o',
    toolsets=[tools],
    capabilities=[IncludeToolReturnSchemas()],
)

# Include return schema only for specific tools
agent_selective = Agent(
    'openai:gpt-4o',
    toolsets=[tools],
    capabilities=[IncludeToolReturnSchemas(tools=['get_weather'])],
)

result = agent_all.run_sync('What is the weather in Tokyo?')
print(result.output)
```

### Filtering by metadata

```python
from pydantic_ai import Agent, FunctionToolset, RunContext, Tool
from pydantic_ai.capabilities import IncludeToolReturnSchemas

tools = FunctionToolset[None]()

@tools.tool_plain
def simple_lookup(key: str) -> str:
    """Simple string lookup."""
    return f'value_for_{key}'

@tools.tool_plain
def structured_fetch(item_id: int) -> dict:
    """Fetch a structured item."""
    return {'id': item_id, 'name': 'Widget', 'price': 9.99}

# Tag tools that should expose return schemas
tools2 = FunctionToolset([
    Tool(simple_lookup),
    Tool(structured_fetch, metadata={'expose_schema': True}),
])

# Only include schemas for tools tagged with expose_schema=True
agent = Agent(
    'openai:gpt-4o',
    toolsets=[tools2],
    capabilities=[IncludeToolReturnSchemas(tools={'expose_schema': True})],
)
result = agent.run_sync('Fetch item 42.')
print(result.output)
```

### Toolset form — `IncludeReturnSchemasToolset`

```python
from pydantic import BaseModel
from pydantic_ai import Agent, FunctionToolset, IncludeReturnSchemasToolset, RunContext

class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str

search_tools = FunctionToolset[None]()

@search_tools.tool_plain
def web_search(query: str) -> list[SearchResult]:
    """Search the web and return top results."""
    return [SearchResult(url='https://example.com', title='Example', snippet='...')]

# Wrapping at toolset level
agent = Agent(
    'openai:gpt-4o',
    toolsets=[IncludeReturnSchemasToolset(search_tools)],
)

result = agent.run_sync('Find Python documentation.')
print(result.output)
```

### Per-tool opt-out

If you've applied `IncludeToolReturnSchemas` or `IncludeReturnSchemasToolset` globally but want to exclude a specific tool, set `include_return_schema=False` on that tool:

```python
from pydantic_ai import Agent, FunctionToolset, Tool, RunContext
from pydantic_ai.capabilities import IncludeToolReturnSchemas

def noop() -> bytes:
    """Returns raw bytes — schema not useful."""
    return b'\xff\xd8\xff'

tools = FunctionToolset([
    Tool(noop, include_return_schema=False),  # ← explicitly excluded
])

agent = Agent(
    'openai:gpt-4o',
    toolsets=[tools],
    capabilities=[IncludeToolReturnSchemas()],   # applies to all other tools
)
```

---

## 9. `NativeOutput`, `PromptedOutput`, `TextOutput` — advanced patterns

Source: `pydantic_ai/output.py`

These marker classes give fine-grained control over how the model returns structured data. For basic usage see `pydantic_ai_output_types.md`; this section covers advanced patterns.

### `NativeOutput` with a custom schema prompt template

When `NativeOutput` is used with a provider that sends the JSON schema as a prompt (e.g. certain Ollama configurations), you can override the template string. Use `{schema}` as the placeholder:

```python
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput

class ExtractionResult(BaseModel):
    entities: list[str]
    sentiment: str
    keywords: list[str]

CUSTOM_TEMPLATE = (
    'Extract structured data from the text.\n'
    'You MUST return JSON conforming exactly to this schema:\n'
    '```json\n{schema}\n```\n'
    'Do not add any fields not in the schema.'
)

agent = Agent(
    'ollama:llama3.2',
    output_type=NativeOutput(
        ExtractionResult,
        template=CUSTOM_TEMPLATE,
        description='Extract entities, sentiment, and keywords.',
    ),
)

result = agent.run_sync('Python is a great language loved by data scientists.')
print(result.output.entities)
print(result.output.sentiment)
```

### `NativeOutput` — disable schema prompt (`template=False`)

When the model profile already injects the schema (e.g. OpenAI's `response_format=json_schema`), passing `template=False` avoids injecting it twice:

```python
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput

class Summary(BaseModel):
    title: str
    points: list[str]
    word_count: int

# OpenAI handles the schema natively — no prompt injection needed
agent = Agent(
    'openai:gpt-4o',
    output_type=NativeOutput(Summary, template=False),
)
result = agent.run_sync('Summarise: Python is a versatile, readable language.')
print(result.output)
```

### `PromptedOutput` — custom schema injection template

`PromptedOutput` injects the schema as text into the system prompt. Override `template` to control the exact phrasing:

```python
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput

class TaskPlan(BaseModel):
    goal: str
    steps: list[str]
    estimated_hours: float

TEMPLATE = (
    '## Output format\n'
    'Respond with **only** a JSON object matching this schema:\n'
    '```\n{schema}\n```\n'
    'No prose, no explanation — only the JSON.'
)

agent = Agent(
    'mistral:mistral-large-latest',
    output_type=PromptedOutput(TaskPlan, template=TEMPLATE),
)

result = agent.run_sync('Plan: build a REST API with authentication.')
print(result.output.goal)
print(result.output.steps[:3])
```

### `TextOutput` with `RunContext` — access deps in the parser

`TextOutputFunc` can optionally take `RunContext` as its first argument, giving your parser access to dependencies, the run ID, and message history:

```python
import asyncio
import re
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext, TextOutput

@dataclass
class ParseDeps:
    currency_symbol: str = '$'

def parse_price(ctx: RunContext[ParseDeps], text: str) -> float:
    """Extract a price from the model's output, using the dep currency symbol."""
    symbol = re.escape(ctx.deps.currency_symbol)
    match = re.search(rf'{symbol}\s*([\d,]+\.?\d*)', text)
    if match:
        return float(match.group(1).replace(',', ''))
    # Fallback: try to extract any number
    numbers = re.findall(r'[\d,]+\.?\d*', text)
    return float(numbers[0].replace(',', '')) if numbers else 0.0

agent = Agent(
    'openai:gpt-4o',
    deps_type=ParseDeps,
    output_type=TextOutput(parse_price),
    instructions='When asked for a price, state it as $X.XX.',
)

async def main():
    result = await agent.run(
        'What is the typical price of a coffee?',
        deps=ParseDeps(currency_symbol='$'),
    )
    price: float = result.output
    print(f'Extracted price: {price:.2f}')

asyncio.run(main())
```

### `TextOutput` async function

The parser can be `async`:

```python
import asyncio
import re
import httpx
from pydantic_ai import Agent, TextOutput

async def enrich_with_lookup(text: str) -> dict:
    """Call an external API to enrich the model's text output."""
    async with httpx.AsyncClient() as client:
        # Use regex to reliably extract URLs regardless of surrounding punctuation
        urls = re.findall(r'https?://\S+', text)
        if urls:
            try:
                resp = await client.get(urls[0], timeout=5)
                return {'text': text, 'url_status': resp.status_code}
            except Exception:
                pass
    return {'text': text, 'url_status': None}

agent = Agent(
    'openai:gpt-4o',
    output_type=TextOutput(enrich_with_lookup),
    instructions='Include a relevant URL in your response.',
)

async def main():
    result = await agent.run('Tell me about the Python docs website.')
    print(result.output)   # dict with 'text' and 'url_status'

asyncio.run(main())
```

### Choosing the right marker class

```
Is your model's profile 'native' or 'tool'?
│
├─ native   → NativeOutput    (best accuracy, no schema in prompt)
├─ tool     → ToolOutput      (explicit named tool for extracting structured output)
├─ prompted → PromptedOutput  (schema injected as text; works everywhere)
└─ you want to post-process text → TextOutput
```

To check what a model defaults to:

```python
from pydantic_ai.profiles import DEFAULT_PROFILE
print(DEFAULT_PROFILE.default_structured_output_mode)  # 'tool'

# Check a specific model
from pydantic_ai.models.openai import OpenAIChatModel
m = OpenAIChatModel('gpt-4o')
print(m.profile.default_structured_output_mode)
```

---

## Quick-reference table

| Class | Module | Key method / attribute | Primary use |
|---|---|---|---|
| `AgentSpec` | `pydantic_ai.agent` | `from_file()`, `from_text()`, `to_file()`, `to_agent()` | YAML/JSON agent config |
| `SkipModelRequest` | `pydantic_ai.exceptions` | `.response: ModelResponse` | Bypass model in hooks |
| `SkipToolValidation` | `pydantic_ai.exceptions` | `.validated_args: dict` | Bypass arg validation |
| `SkipToolExecution` | `pydantic_ai.exceptions` | `.result: Any` | Bypass tool execution |
| `ProcessEventStream` | `pydantic_ai.capabilities` | `.handler` (observer or processor) | Monitor / transform event stream |
| `HandleDeferredToolCalls` | `pydantic_ai.capabilities` | `.handler(ctx, requests) -> results` | Auto-resolve deferred/approval calls |
| `PreparedToolset` | `pydantic_ai.toolsets` | `.prepare_func(ctx, defs) -> defs` | Dynamic tool definition mutation |
| `DeferredLoadingToolset` | `pydantic_ai.toolsets` | `.tool_names` (frozenset or None) | Hide tools until searched |
| `InstructionPart` | `pydantic_ai.messages` | `.dynamic`, `.join()`, `.sorted()` | Cacheable instruction composition |
| `IncludeToolReturnSchemas` | `pydantic_ai.capabilities` | `.tools` selector | Add return schemas (capability) |
| `IncludeReturnSchemasToolset` | `pydantic_ai.toolsets` | `__init__(wrapped)` | Add return schemas (toolset) |
| `NativeOutput` | `pydantic_ai` | `.outputs`, `.template`, `.strict` | Provider-native structured output |
| `PromptedOutput` | `pydantic_ai` | `.outputs`, `.template` | Prompt-guided structured output |
| `TextOutput` | `pydantic_ai` | `.output_function(text)` | Post-process raw text output |

---
title: "Mistral Agents API: Comprehensive Technical Guide"
description: "Reference for Mistral's Agents and Conversations APIs with the mistralai Python SDK: agents, versions, conversations, streaming, tools, handoffs, structured output, guardrails and errors."
framework: "mistral-agents-api"
language: "python"
sidebar:
  label: "Core reference"
  order: 1
---

**Verified against `mistralai` %%version=mistralai%%** (checked %%versions-checked%%). Every method, parameter and tool shape on this page was checked against an installed copy of the SDK. Diagrams are on the [diagrams page](/mistral-agents-api-guide/mistral_agents_api_diagrams/).

**Status:** the Agents and Conversations APIs are beta (they live under `client.beta`), and registered MCP connectors are a public preview. Request and response shapes can still change between SDK releases.

---

## Contents

1. [Overview](#1-overview)
2. [Installation and authentication](#2-installation-and-authentication)
3. [Models](#3-models)
4. [Agents](#4-agents)
5. [Conversations](#5-conversations)
6. [Streaming](#6-streaming)
7. [Built-in tools](#7-built-in-tools)
8. [Function calling](#8-function-calling)
9. [Handoffs](#9-handoffs)
10. [Structured output](#10-structured-output)
11. [Guardrails](#11-guardrails)
12. [Errors, retries and timeouts](#12-errors-retries-and-timeouts)
13. [Async usage](#13-async-usage)
14. [Changes in SDK v3](#14-changes-in-sdk-v3)

---

## 1. Overview

The Agents API has two halves:

- **Agents** (`client.beta.agents`) — stored, versioned configurations: model, instructions, tools, completion arguments, guardrails and the agents this one may hand off to.
- **Conversations** (`client.beta.conversations`) — server-side conversation state. You start a conversation with an agent (or directly with a model), then append to it, restart it from an earlier entry, or read its history. You never resend the full transcript.

Built-in tools (web search, code interpreter, image generation, document libraries and MCP connectors) run on Mistral's side. Your own functions run in your process, either through a manual loop or with the SDK's `RunContext` helper.

| Concept | Where it lives |
|---|---|
| Agent configuration and versions | Mistral (`/v1/agents`) |
| Conversation history | Mistral (`/v1/conversations`), unless `store=False` |
| Built-in tool execution | Mistral |
| Function tools | your code |
| MCP servers | Mistral (registered connectors) or your code (`mistralai.extra.mcp`) |

Related Mistral features that this guide does not cover in depth:

- **Le Chat deployment.** `agents.update(..., deployment_chat=True)` makes an agent available in Le Chat; `agents.list_pages(deployment_chat=True)` lists those agents.
- **Reasoning effort.** `completion_args={"reasoning_effort": "low"}` on models that support it. The SDK accepts `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` and, from 3.5.0, `"max"` (see [below](#added-in-350-max-reasoning-effort)).
- **Prompts and Skills** (`client.beta.prompts`, `client.beta.skills`): stored, versioned prompt and skill objects.
- **Workflows** (`client.workflows`, `mistralai.extra.workflows`) and **voice agents** (`mistralai.extra.voice_agents`) are separate products with their own documentation.

---

## 2. Installation and authentication

Python 3.10 or later.

```bash
pip install "mistralai==%%version=mistralai%%"
# with RunContext, MCP clients and their dependencies:
pip install "mistralai[agents]==%%version=mistralai%%"
# only the official MCP client (connectors.mcp_client()), without RunContext's extras:
pip install "mistralai[mcp]==%%version=mistralai%%"
```

Import the client from `mistralai.client` (the top-level `mistralai` package is a namespace and exports nothing):

```python
import os
from mistralai.client import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
```

`Mistral()` with no arguments resolves credentials in this order: an `Authorization` header you set yourself, an explicit `api_key`, the service-account token file named by `MISTRAL_SA_TOKEN_PATH`, then `MISTRAL_API_KEY`. If both `MISTRAL_SA_TOKEN_PATH` and `MISTRAL_API_KEY` are set, the token file wins. An `api_key` passed to the constructor outranks both, so `Mistral(api_key=os.environ["MISTRAL_API_KEY"])`, as used in most examples, never reads a token file; call `Mistral()` with no key to get the order above. The token file is read on every request, and a missing or empty file raises `ServiceAccountTokenError`.

Use the client as a context manager in scripts so its HTTP connections are closed:

```python
from mistralai.client import Mistral

with Mistral() as client:
    print([m.id for m in client.models.list().data][:5])
```

Region-specific endpoints are available as `SERVER_EU` and `SERVER_US` (`Mistral(server="eu")`). Azure and Google Cloud deployments use `mistralai.azure.client.MistralAzure` and `mistralai.gcp.client.MistralGCP`.

---

## 3. Models

Agents accept any chat model your workspace can use. Current general-purpose choices, from Mistral's [models overview](https://docs.mistral.ai/models):

| Model | API name | Alias | Context | Use for |
|---|---|---|---|---|
| Mistral Medium 3.5 | `mistral-medium-3-5` | `mistral-medium-latest` | 256k | Default for agentic and coding work |
| Mistral Large 3 | `mistral-large-2512` | `mistral-large-latest` | 256k | Open-weight flagship |
| Mistral Large 4 (preview) | `mistral-large-4` | — | 1M | Very long contexts; public preview |
| Mistral Small 4 | `mistral-small-2603` | `mistral-small-latest` | 256k | Fast, cheap agents and routers |
| Ministral 3 8B | `ministral-8b-2512` | — | 256k | Edge and high-volume classification |

API names and context windows come from the model pages on docs.mistral.ai (checked %%versions-checked%%). Those pages do not state which model each `-latest` alias currently points to, so treat the alias column as the usual convention and confirm it on your account (below). Aliases move when Mistral releases a new version, so pin the versioned API name (`mistral-medium-3-5`, `mistral-small-2603`) in production and use aliases in prototypes. `mistral-medium-2505` (Medium 3) and `mistral-medium-2508` (Medium 3.1) were deprecated on 22 May 2026 and retired on 31 August 2026; their replacement is Mistral Medium 3.5.

Each model card returned by the API carries its `aliases`, `max_context_length` and, for models on their way out, `deprecation` and `deprecation_replacement_model`, so a deploy script can check what an alias resolves to:

```python
from mistralai.client import Mistral

wanted = {"%%model=mistral%%", "%%model=mistral-small%%"}

with Mistral() as client:
    for card in client.models.list().data:
        if card.id in wanted or wanted & set(card.aliases or []):
            print(card.id, card.aliases, card.max_context_length, card.deprecation)
```

---

## 4. Agents

### Create

```python
import os
from mistralai.client import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

agent = client.beta.agents.create(
    model="%%model=mistral%%",
    name="research-assistant",
    description="Answers questions using current information from the web.",
    instructions="Search before answering anything time-sensitive. Cite sources inline.",
    tools=[{"type": "web_search"}],
    completion_args={"temperature": 0.3, "top_p": 0.95},
    metadata={"team": "docs"},
)
print(agent.id, agent.version)
```

`create()` takes `model`, `name`, and optionally `instructions`, `tools`, `completion_args`, `guardrails`, `description`, `handoffs`, `metadata` and `version_message`. All SDK methods are keyword-only.

### Read, list, delete

```python
agent = client.beta.agents.get(agent_id=agent.id)

page = client.beta.agents.list_pages(page_size=50, name="research-assistant")
while page is not None:
    for a in page.result.data:
        print(a.id, a.name, a.version)
    page = page.next()          # None after the last page

client.beta.agents.delete(agent_id=agent.id)
```

`list_pages()` is cursor-paginated (`page.result.next_page_token`, or pass `page_token=` yourself) and filters by `name`, `search`, `metadata` and `deployment_chat`. The older `agents.list()` is deprecated: it ignores agent sharing and emits a `DeprecationWarning`.

### Versions and aliases

Every `update()` creates a new version. Conversations pin to whichever version was current when they started, unless you ask for a specific one.

```python
agent = client.beta.agents.update(
    agent_id=agent.id,
    instructions="Search before answering. Cite sources as [n] with a list at the end.",
    version_message="Numbered citations",
)

versions = client.beta.agents.list_versions(agent_id=agent.id)
client.beta.agents.update_version(agent_id=agent.id, version=0)        # roll back
client.beta.agents.create_version_alias(agent_id=agent.id, alias="prod", version=1)

# Start a conversation on a specific version or alias
response = client.beta.conversations.start(
    agent_id=agent.id, agent_version="prod", inputs="What changed in EU AI rules this month?"
)
```

Other version calls:

```python
client.beta.agents.get_version(agent_id=agent.id, version="1")         # one version (version is a string here)
client.beta.agents.list_version_aliases(agent_id=agent.id)             # every alias and the version it points to
client.beta.agents.delete_version_alias(agent_id=agent.id, alias="staging")
```

`list_versions()` is offset-paginated (`page`, `page_size`, default 20) and returns a plain list of `Agent` objects. Note that `update_version()` takes an integer `version`, whereas `get_version()` takes a string.

---

## 5. Conversations

### Start

```python
response = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="Who won the 2026 Tour de France?",
)
print(response.conversation_id)
for entry in response.outputs:
    print(entry.type)          # e.g. "tool.execution", then "message.output"
```

`inputs` is a string or a list of entries such as `{"role": "user", "content": "..."}`. Without an agent, pass `model=` (and optionally `instructions=`, `tools=` and `completion_args=`) instead of `agent_id=`; the conversation then carries its own configuration and nothing is stored as an agent. Use this for ad-hoc work or while you are still iterating on a prompt, and move to a stored agent when several callers should share one definition and one version history. `agent_id` and `model` are alternatives: send one of them.

`append()` also takes `completion_args=` to change sampling settings for one turn (for example a lower `temperature` or a `max_tokens` cap) without touching the agent.

`outputs` can contain tool executions, function calls, handoffs and messages. The answer is in the last `message.output` entry, whose `content` is either a string or a list of chunks (text, tool references for citations, files). A helper used throughout these pages:

```python
from mistralai.client.models import ConversationResponse, MessageOutputEntry, TextChunk

def text_of(response: ConversationResponse) -> str:
    """Return the text of the last message in a conversation response."""
    for entry in reversed(response.outputs):
        if isinstance(entry, MessageOutputEntry):
            if isinstance(entry.content, str):
                return entry.content
            return "".join(c.text for c in entry.content if isinstance(c, TextChunk))
    return ""
```

### Continue

```python
response = client.beta.conversations.append(
    conversation_id=response.conversation_id,
    inputs="And who came second?",
)
print(text_of(response))
```

`append()` continues with the same agent. To bring in a different agent, use [handoffs](#9-handoffs) or start a new conversation with the other agent and pass it the context it needs.

### Restart from an earlier entry

```python
history = client.beta.conversations.get_history(conversation_id=response.conversation_id)
first_user_entry = history.entries[0]

branch = client.beta.conversations.restart(
    conversation_id=response.conversation_id,
    from_entry_id=first_user_entry.id,
    inputs="Answer again, but only for the women's race.",
)
```

`restart()` creates a new conversation that shares history up to `from_entry_id`.

### Read and manage

```python
conv = client.beta.conversations.get(conversation_id=response.conversation_id)
messages = client.beta.conversations.get_messages(conversation_id=response.conversation_id)
recent = client.beta.conversations.list(page_size=20, metadata={"team": "docs"})
client.beta.conversations.delete(conversation_id=response.conversation_id)
```

Pass `store=False` to `start()` for one-off requests that should not be persisted (they cannot be appended to afterwards).

### Usage

```python
usage = response.usage
print(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, usage.connector_tokens)
```

---

## 6. Streaming

Every conversation method has a `_stream` variant that returns server-sent events:

```python
with client.beta.conversations.start_stream(
    agent_id=agent.id,
    inputs="Summarise this week's ECB announcements.",
) as stream:
    for event in stream:
        data = event.data
        if data.type == "message.output.delta" and isinstance(data.content, str):
            print(data.content, end="", flush=True)
        elif data.type == "tool.execution.started":
            print(f"\n[tool: {data.name}]")
        elif data.type == "conversation.response.done":
            print(f"\n[{data.usage.total_tokens} tokens]")
```

Event types: `conversation.response.started`, `message.output.delta`, `tool.execution.started` / `.delta` / `.done`, `function.call.delta`, `agent.handoff.started` / `.done`, `conversation.response.done` and `conversation.response.error`. `append_stream()` and `restart_stream()` behave the same way.

---

## 7. Built-in tools

| `type` | What it does | Extra fields |
|---|---|---|
| `web_search` | Web search with citations | `tool_configuration` |
| `web_search_premium` | Web search plus news agencies | `tool_configuration` |
| `code_interpreter` | Runs Python in a sandbox | `tool_configuration` |
| `image_generation` | Generates images (returned as files) | `tool_configuration` |
| `document_library` | Retrieval over your Libraries | `library_ids`, `tool_configuration` |
| `connector` | A registered MCP connector | `connector_id`, `authorization`, `tool_configuration` |
| `function` | Your own function | `function` |

Since SDK v3 these tools are only accepted by the Conversations and Agents APIs: `chat.complete()` and `agents.complete()` no longer take `web_search`, `web_search_premium` or `code_interpreter`. See the [connectors guide](/mistral-agents-api-guide/mistral_agents_api_connectors_guide/) for each tool in detail.

---

## 8. Function calling

### Manual loop

Declare the function on the agent; when the model calls it, the response ends with a `function.call` entry, and you append a `FunctionResultEntry`:

```python
import json
from mistralai.client.models import FunctionCallEntry, FunctionResultEntry

def get_order_status(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped", "eta": "2026-10-24"}

support = client.beta.agents.create(
    model="%%model=mistral-small%%",
    name="order-support",
    instructions="Use get_order_status for any question about an order.",
    tools=[{
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the status of an order by its ID.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "e.g. ORD-1234"}},
                "required": ["order_id"],
            },
        },
    }],
)

response = client.beta.conversations.start(agent_id=support.id, inputs="Where is ORD-1234?")
while isinstance(response.outputs[-1], FunctionCallEntry):
    call = response.outputs[-1]
    args = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
    result = get_order_status(**args)
    response = client.beta.conversations.append(
        conversation_id=response.conversation_id,
        inputs=[FunctionResultEntry(tool_call_id=call.tool_call_id, result=json.dumps(result))],
    )
print(text_of(response))
```

### `RunContext`

With `pip install "mistralai[agents]"`, `RunContext` builds the tool schema from a Python function's signature and docstring, executes calls, and loops until the model answers:

```python
import asyncio
import os
from mistralai.client import Mistral
from mistralai.extra.run.context import RunContext

def get_order_status(order_id: str) -> dict:
    """Look up the status of an order.

    Args:
        order_id: The order ID, e.g. ORD-1234.
    """
    return {"order_id": order_id, "status": "shipped"}

async def main() -> None:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    async with RunContext(model="%%model=mistral-small%%") as run_ctx:
        run_ctx.register_func(get_order_status)
        result = await client.beta.conversations.run_async(
            run_ctx=run_ctx,
            inputs="Where is ORD-1234?",
            instructions="Use the tools to answer order questions.",
        )
        print(result.output_as_text)
        print("conversation:", result.conversation_id)

asyncio.run(main())
```

`register_func(fn, requires_confirmation=True)` pauses the run with a `DeferredToolCallsException` so a human can approve the call; see the [MCP guide](/mistral-agents-api-guide/mistral_agents_api_mcp_guide/#3-human-approval-for-tool-calls). `RunContext(agent_id=...)` runs against a stored agent instead of a model; `model` and `agent_id` cannot both be set.

Behaviour to know before relying on it, all read from the installed SDK:

- **There is no built-in turn limit.** `run_async()` keeps appending function results until the model answers without calling a function. A model that keeps calling tools keeps costing you; check `run_ctx.request_count` or run the manual loop above with a counter when that matters.
- **A function that raises stops the run.** The exception is wrapped in `mistralai.extra.exceptions.RunException`. With `RunContext(..., continue_on_fn_error=True)` the error text (`Error while executing <name>: ...`) is sent back to the model as the function result instead, so it can retry or explain.
- **Functions can be sync or `async def`.** Synchronous functions run on the event loop's thread, so a slow blocking call stalls other tasks; use an `async def` (or run blocking work in an executor) for I/O.
- **Return values** are sent as strings: a `str` as is, anything else through `json.dumps`.
- `result.entries` holds the input and output entries, `result.output_as_text` joins the assistant's messages, and `result.get_file(file_id)` returns generated files that were fetched during the run.

`run_stream_async()` is the streaming form. It is a coroutine that returns an async generator, so `await` it first; it yields the same events as `start_stream()`, and the last item is the `RunResult`:

```python
import asyncio
import os
from mistralai.client import Mistral
from mistralai.extra.run.context import RunContext
from mistralai.extra.run.result import RunResult

def lookup_price(sku: str) -> float:
    """Return the unit price of a SKU in euros.

    Args:
        sku: Product code, e.g. SKU-1.
    """
    prices = {"SKU-1": 12.5}
    return prices[sku]          # raises KeyError for an unknown SKU

async def main() -> None:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    async with RunContext(model="%%model=mistral-small%%", continue_on_fn_error=True) as run_ctx:
        run_ctx.register_func(lookup_price)
        stream = await client.beta.conversations.run_stream_async(
            run_ctx=run_ctx,
            inputs="What do SKU-1 and SKU-9 cost?",
            instructions="Use lookup_price for every SKU. Say so if a lookup fails.",
        )
        async for item in stream:
            if isinstance(item, RunResult):          # the last item summarises the run
                print("\nrequests made:", run_ctx.request_count)
                print("conversation:", item.conversation_id)
            elif item.data.type == "message.output.delta" and isinstance(item.data.content, str):
                print(item.data.content, end="", flush=True)
            elif item.data.type == "function.call.delta":
                print(f"\n[calling {item.data.name}({item.data.arguments})]")

asyncio.run(main())
```

Here the `SKU-9` lookup fails with a `KeyError`; because of `continue_on_fn_error=True` the model receives the error text and can say so, where without it the run would raise `RunException`.

---

## 9. Handoffs

An agent can hand the conversation to other agents listed in its `handoffs`:

```python
billing = client.beta.agents.create(
    model="%%model=mistral-small%%", name="billing",
    description="Handles invoices, refunds and payment questions.",
    instructions="You resolve billing questions.",
)
tech = client.beta.agents.create(
    model="%%model=mistral%%", name="tech-support",
    description="Handles technical problems with the product.",
    instructions="You troubleshoot technical issues step by step.",
    tools=[{"type": "web_search"}],
)
triage = client.beta.agents.create(
    model="%%model=mistral-small%%", name="triage",
    description="Routes customer requests.",
    instructions="Decide which specialist should answer and hand off. Do not answer yourself.",
    handoffs=[billing.id, tech.id],
)

response = client.beta.conversations.start(
    agent_id=triage.id,
    inputs="I was charged twice this month.",
    handoff_execution="server",   # the default: the platform runs the target agent in the same request
)
for entry in response.outputs:
    if entry.type == "agent.handoff":
        print("handed off to", entry.next_agent_name)
print(text_of(response))
```

With `handoff_execution="client"` the response stops at the handoff so your code can log, check or modify state before continuing. The [orchestration guide](/mistral-agents-api-guide/mistral_agents_api_orchestration_guide/) covers handoff graphs and code-driven pipelines.

---

## 10. Structured output

Set a JSON schema through `completion_args.response_format`:

```python
from pydantic import BaseModel
from mistralai.extra import response_format_from_pydantic_model

class Ticket(BaseModel):
    category: str
    priority: int
    summary: str

classifier = client.beta.agents.create(
    model="%%model=mistral-small%%",
    name="ticket-classifier",
    instructions="Classify the support ticket.",
    completion_args={"response_format": response_format_from_pydantic_model(Ticket)},
)

response = client.beta.conversations.start(agent_id=classifier.id, inputs="The app crashes on login since today.")
ticket = Ticket.model_validate_json(text_of(response))
```

`RunContext(output_format=Ticket)` does the same and exposes `result.output_as_model`. For a single call without an agent, `client.chat.parse(response_format=Ticket, model=..., messages=[...])` returns a parsed object.

---

## 11. Guardrails

Attach Mistral's moderation models to an agent or a single conversation:

```python
from mistralai.client.models import GuardrailConfig, ModerationLlmv2Config

safe = client.beta.agents.create(
    model="%%model=mistral%%",
    name="public-assistant",
    instructions="You are a helpful assistant for the public website.",
    guardrails=[GuardrailConfig(block_on_error=True, moderation_llm_v2=ModerationLlmv2Config())],
)
```

`ModerationLlmv2Config` uses `mistral-moderation-2603` (Mistral Moderation 2) and accepts `custom_category_thresholds`, `ignore_other_categories` and `action`. `ModerationLlmv1Config` uses the deprecated `mistral-moderation-2411`, which reached retirement on 30 June 2026; use v2. `block_on_error=True` makes the API return HTTP 403 and block the request when a server-side error occurs while moderating (the default, `False`, lets the request through). `action` is `"none"` or `"block"`. Guardrail results are reported on `response.guardrails`, a list of free-form dictionaries, so log them rather than relying on a fixed shape.

To act on only some categories, give thresholds and ignore the rest. The categories are `sexual`, `hate_and_discrimination`, `violence_and_threats`, `dangerous`, `criminal`, `selfharm`, `health`, `financial`, `law`, `pii` and `jailbreaking`, each a float:

```python
from mistralai.client.models import GuardrailConfig, ModerationLlmv2Config
from mistralai.client.models import ModerationLlmv2CategoryThresholds

strict_on_pii = GuardrailConfig(
    block_on_error=True,
    moderation_llm_v2=ModerationLlmv2Config(
        custom_category_thresholds=ModerationLlmv2CategoryThresholds(pii=0.5, jailbreaking=0.5),
        ignore_other_categories=True,
        action="block",
    ),
)

response = client.beta.conversations.start(
    model="%%model=mistral-small%%",
    inputs="Summarise our refund policy.",
    guardrails=[strict_on_pii],      # applies to this conversation only
)
```

The SDK does not document how a score maps to a threshold; the values above are placeholders to tune against Mistral's moderation documentation and your own traffic.

---

## 12. Errors, retries and timeouts

HTTP errors raise subclasses of `mistralai.client.errors.MistralError`, which carry `status_code`, `body`, `headers` (useful for `Retry-After`) and `raw_response`, the `httpx2` response. `HTTPValidationError` is raised for 422s and `SDKError` for other error statuses. `ResponseValidationError` means a response did not match the SDK's model (often an SDK that is older than the API), and `ObservabilityError` comes from the observability endpoints. `NoResponseError` (no response arrived) is the exception: it derives from `Exception`, not `MistralError`, so an `except MistralError` clause does not catch it. Transport failures such as timeouts surface as `httpx2` exceptions.

```python
from mistralai.client import Mistral, errors
from mistralai.client.utils import BackoffStrategy, RetryConfig

client = Mistral(
    retry_config=RetryConfig(
        "backoff",
        BackoffStrategy(initial_interval=500, max_interval=30_000, exponent=1.8, max_elapsed_time=120_000),
        retry_connection_errors=True,
    ),
    timeout_ms=60_000,
)

try:
    response = client.beta.conversations.start(agent_id="ag_missing", inputs="Hello")
except errors.HTTPValidationError as e:
    print("bad request:", e.data)
except errors.MistralError as e:
    if e.status_code == 429:
        print("rate limited; retry later")
    elif e.status_code == 404:
        print("no such agent")
    else:
        raise
```

`retries=`, `timeout_ms=` and `http_headers=` can also be passed per call.

---

## 13. Async usage

Every method has an `_async` twin:

```python
import asyncio
from mistralai.client import Mistral

async def ask_all(agent_id: str, questions: list[str]) -> list[str]:
    async with Mistral() as client:
        responses = await asyncio.gather(*(
            client.beta.conversations.start_async(agent_id=agent_id, inputs=q) for q in questions
        ))
        return [text_of(r) for r in responses]
```

For streaming: `async with await client.beta.conversations.start_stream_async(...) as stream: async for event in stream: ...`.

---

## 14. Changes in SDK v3

`mistralai` 3.0.0 (28 September 2026) kept imports and method names from v2 (`from mistralai.client import Mistral`, `client.beta.agents.create`, `client.beta.conversations.start`). What changed, checked against the installed 3.x package (the upstream [migration guide](https://github.com/mistralai/client-python/blob/main/MIGRATION.md) is linked from the SDK's README):

- **HTTP transport is `httpx2`.** Custom clients must be `httpx2.Client` / `httpx2.AsyncClient`; catch `httpx2` exceptions, not `httpx` ones.
- **`mcp` 2.2.** The `agents` extra depends on `mcp>=2.2,<3`; MCP results use 2.x field names (`structured_content`, not `structuredContent`).
- **Service-account tokens.** `MISTRAL_SA_TOKEN_PATH` is read automatically and takes precedence over `MISTRAL_API_KEY`.
- **Built-in tools left chat completions.** `web_search`, `web_search_premium` and `code_interpreter` are no longer accepted by `chat.complete`, `chat.stream`, `agents.complete` or `agents.stream`; use Conversations or an agent.
- **Connectors (beta).** `delete_user_credentials` / `delete_workspace_credentials` / `delete_organization_credentials` became `delete_credentials(..., consumer_scope=...)`; `list_tools()` lost `page`.
- **Other beta areas moved too** (observability, workflows). They are outside this guide; read the SDK's `MIGRATION.md` before upgrading code that uses them.

### Added in 3.4.0: the stateless Responses API

`mistralai` 3.4.0 (23 October 2026) added `client.responses`, an OpenAI-style Responses endpoint (`POST /v1/responses`). It is separate from agents and conversations: the SDK describes it as stateless, so you send the full history in `input` each time, nothing is stored, and only client-executed function tools are supported. Use it for plain model calls with typed output items; use Conversations when you want hosted state, hosted tools or handoffs. The Python and TypeScript (3.2.0) clients are otherwise unchanged for agents and conversations.

```python
from mistralai.client import Mistral

client = Mistral(api_key="...")

response = client.responses.create(
    model="%%model=mistral%%",
    input="What is the capital of France?",
    instructions="Answer in one word.",
)
message = response.output[0]            # a "message" item; reasoning and function-call items also appear in output
print(message.content[0].text)
print(client.responses.count_input_tokens(model="%%model=mistral%%", input="What is the capital of France?").input_tokens)
```

We ran this against a scripted `httpx2.MockTransport`, not the live API, so the output shape is the SDK's model rather than a recorded reply. `stream=True` returns server-sent events, and `previous_response_id`, `conversation` and `prompt` are accepted by the signature but we did not exercise them. The TypeScript equivalent is `await client.responses.create({ model, input, instructions })`.

Two smaller 3.4.0 changes sit outside this guide: `audio_agents.create()` now requires `configuration` (it was optional), and `KeywordRetriever` gained an `operator` field (`"or"` by default, or `"and"`).

### Added in 3.5.0: `max` reasoning effort

`mistralai` 3.5.0 (25 October 2026) added one value, `"max"`, to the `ReasoningEffort` type. It is accepted wherever reasoning effort is: an agent's or a conversation's `completion_args`, and the `reasoning_effort=` argument of `chat.complete`, `chat.stream`, `agents.complete` and `agents.stream`. Nothing else in the SDK changed. The TypeScript client gained the same value in 3.3.0 (`completionArgs: { reasoningEffort: "max" }`).

```python
from mistralai.client import Mistral

client = Mistral(api_key="...")

agent = client.beta.agents.create(
    model="%%model=mistral%%",
    name="deep-reviewer",
    instructions="Review the design carefully before answering.",
    completion_args={"reasoning_effort": "max"},
)
```

We checked this against a scripted `httpx2.MockTransport`: the SDK sends `"reasoning_effort": "max"` and reads it back on the returned agent. Which models accept `"max"`, and how it differs from `"xhigh"` in cost and latency, is decided by the API; we have not tested it against the live service, so check the model's page on docs.mistral.ai first. Upgrade before you rely on it: in the same mock test `mistralai` 3.4.0 accepted `"max"` without an error but left it out of the request body, so the call would run at the model's default effort. The TypeScript 3.2.0 client, by contrast, sent the string through unchanged.

Upgrading from v1 as well? v2 moved every import from `mistralai` to `mistralai.client`, and Azure/GCP clients to `mistralai.azure.client` / `mistralai.gcp.client`.

### Earlier history

- **27 May 2025.** Mistral launched the Agents API as a beta: stored agents, server-side conversations (append, restart, streaming), built-in code execution, image generation, web search (with a premium news variant) and document libraries, handoffs, and MCP tool use. The launch post lists "persistent memory across conversations" without describing a mechanism beyond stored, branchable conversation history; this guide treats it as that history, not as a separate tool.
- **2025–2026.** `document_library` with `library_ids`; MCP `connector` tools with stored credentials and OAuth; `tool_configuration` (`include`, `exclude`, `requires_confirmation`); agent versions and aliases; guardrails; `handoff_execution`.
- **March 2026.** Python SDK 2.0.0 (10 March) moved imports to `mistralai.client` and the Azure and GCP clients to `mistralai.azure.client` and `mistralai.gcp.client`.
- **2026 model retirements.** `mistral-moderation-2411` retired on 30 June 2026 (replaced by `mistral-moderation-2603`); `mistral-medium-2505` and `mistral-medium-2508` retired on 31 August 2026 (replaced by `mistral-medium-3-5`). Dates are from Mistral's model-deprecation table.

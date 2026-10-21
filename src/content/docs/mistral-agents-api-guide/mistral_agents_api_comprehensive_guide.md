---
title: "Mistral Agents API: Comprehensive Technical Guide"
description: "Reference for Mistral's Agents and Conversations APIs with the mistralai Python SDK: agents, versions, conversations, streaming, tools, handoffs, structured output, guardrails and errors."
framework: mistral-agents-api
---

# Mistral Agents API: Comprehensive Technical Guide

**Verified against `mistralai` %%version=mistralai%%** (checked %%versions-checked%%). Every method, parameter and tool shape on this page was checked against an installed copy of the SDK. The architecture diagrams that used to open this page now live on the [diagrams page](/mistral-agents-api-guide/mistral_agents_api_diagrams/).

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

---

## 2. Installation and authentication

Python 3.10 or later.

```bash
pip install "mistralai==%%version=mistralai%%"
# with RunContext, MCP clients and their dependencies:
pip install "mistralai[agents]==%%version=mistralai%%"
```

Import the client from `mistralai.client` (the top-level `mistralai` package is a namespace and exports nothing):

```python
import os
from mistralai.client import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
```

`Mistral()` with no arguments resolves credentials in this order: an `Authorization` header you set yourself, an explicit `api_key`, the service-account token file named by `MISTRAL_SA_TOKEN_PATH`, then `MISTRAL_API_KEY`. If both `MISTRAL_SA_TOKEN_PATH` and `MISTRAL_API_KEY` are set, the token file wins.

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

Aliases move when Mistral releases a new version, so pin the dated API name in production and use aliases in prototypes. `mistral-medium-2505` (Medium 3) and `mistral-medium-2508` (Medium 3.1) reached retirement on 31 August 2026; their replacement is Mistral Medium 3.5.

---

## 4. Agents

### Create

```python
import os
from mistralai.client import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

agent = client.beta.agents.create(
    model="mistral-medium-latest",
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
for a in client.beta.agents.list(page_size=50, name="research-assistant"):
    print(a.id, a.name, a.version)
client.beta.agents.delete(agent_id=agent.id)
```

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

`inputs` is a string or a list of entries such as `{"role": "user", "content": "..."}`. Without an agent, pass `model=` (and optionally `instructions=` and `tools=`) instead of `agent_id=`.

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
| `document_library` | Retrieval over your Libraries | `library_ids` |
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
    model="mistral-small-latest",
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
    async with RunContext(model="mistral-small-latest") as run_ctx:
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

`register_func(fn, requires_confirmation=True)` pauses the run with a `DeferredToolCallsException` so a human can approve the call; see the [MCP guide](/mistral-agents-api-guide/mistral_agents_api_mcp_guide/#3-human-approval-for-tool-calls). `RunContext(agent_id=...)` runs against a stored agent instead of a model.

---

## 9. Handoffs

An agent can hand the conversation to other agents listed in its `handoffs`:

```python
billing = client.beta.agents.create(
    model="mistral-small-latest", name="billing",
    description="Handles invoices, refunds and payment questions.",
    instructions="You resolve billing questions.",
)
tech = client.beta.agents.create(
    model="mistral-medium-latest", name="tech-support",
    description="Handles technical problems with the product.",
    instructions="You troubleshoot technical issues step by step.",
    tools=[{"type": "web_search"}],
)
triage = client.beta.agents.create(
    model="mistral-small-latest", name="triage",
    description="Routes customer requests.",
    instructions="Decide which specialist should answer and hand off. Do not answer yourself.",
    handoffs=[billing.id, tech.id],
)

response = client.beta.conversations.start(
    agent_id=triage.id,
    inputs="I was charged twice this month.",
    handoff_execution="server",   # the platform runs the target agent in the same request
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
    model="mistral-small-latest",
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
    model="mistral-medium-latest",
    name="public-assistant",
    instructions="You are a helpful assistant for the public website.",
    guardrails=[GuardrailConfig(block_on_error=True, moderation_llm_v2=ModerationLlmv2Config())],
)
```

`ModerationLlmv2Config` uses `mistral-moderation-2603` (Mistral Moderation 2) and accepts `custom_category_thresholds`, `ignore_other_categories` and `action`. `ModerationLlmv1Config` uses the deprecated `mistral-moderation-2411`, which reached retirement on 30 June 2026; use v2. `block_on_error=True` rejects the request if moderation itself fails. Guardrail results are reported on `response.guardrails`.

---

## 12. Errors, retries and timeouts

HTTP errors raise subclasses of `mistralai.client.errors.MistralError`, which carry `status_code`, `body` and the raw `httpx2` response. `HTTPValidationError` is raised for 422s; `SDKError` for everything else.

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

`mistralai` 3.0.0 (September 2026) kept imports and method names from v2 (`from mistralai.client import Mistral`, `client.beta.agents.create`, `client.beta.conversations.start`). What changed, from the upstream [migration guide](https://github.com/mistralai/client-python/blob/main/MIGRATION.md):

- **HTTP transport is `httpx2`.** Custom clients must be `httpx2.Client` / `httpx2.AsyncClient`; catch `httpx2` exceptions, not `httpx` ones.
- **`mcp` 2.2.** The `agents` extra depends on `mcp>=2.2,<3`; MCP results use 2.x field names (`structured_content`, not `structuredContent`).
- **Service-account tokens.** `MISTRAL_SA_TOKEN_PATH` is read automatically and takes precedence over `MISTRAL_API_KEY`.
- **Built-in tools left chat completions.** `web_search`, `web_search_premium` and `code_interpreter` are no longer accepted by `chat.complete`, `chat.stream`, `agents.complete` or `agents.stream`; use Conversations or an agent.
- **Connectors (beta).** `delete_user_credentials` / `delete_workspace_credentials` / `delete_organization_credentials` became `delete_credentials(..., consumer_scope=...)`; `list_tools()` lost `page`.
- **Observability and workflow deployment** APIs changed shape (single pipeline `definition`; Mistral Cloud deployment backend).

Upgrading from v1 as well? v2 moved every import from `mistralai` to `mistralai.client`, and Azure/GCP clients to `mistralai.azure.client` / `mistralai.gcp.client`.

---
title: "Microsoft Agent Framework (Python) — Model Providers"
description: "Real imports and constructors for every first-party chat client in agent-framework 1.5.0: OpenAI, Azure OpenAI, Microsoft Foundry, Foundry Local, Anthropic, Ollama, Bedrock, GitHub Copilot, Copilot Studio."
framework: microsoft-agent-framework
language: python
---

# Model Providers — Python

Every chat client in `agent-framework` implements the same `SupportsChatGetResponse` protocol, so `Agent(client=...)` accepts them interchangeably. The import is always `agent_framework.<provider>.<ClassName>` — **no Azure SDK import is required for any of these**. The Azure SDK only becomes relevant for authentication (`azure-identity`) or for Azure-specific storage providers.

This page was verified against `agent-framework-core==1.5.0` and provider packages at `1.0.0b260514` (May 2026). Each sub-package is imported lazily from the `agent_framework.<provider>` namespace — you install the provider package and import from `agent_framework.<provider>`.

## Provider index

| Provider | Package | Import path | Status |
|---|---|---|---|
| OpenAI | `agent-framework-openai` | `agent_framework.openai` | Stable |
| Azure OpenAI | `agent-framework-openai` | `agent_framework.openai` (same client) | Stable |
| Microsoft Foundry | `agent-framework-foundry` | `agent_framework.foundry` | Stable |
| Foundry Local | `agent-framework-foundry-local` | `agent_framework.foundry` | Beta |
| Anthropic | `agent-framework-anthropic` | `agent_framework.anthropic` | Beta |
| Anthropic on Bedrock | `agent-framework-anthropic` | `agent_framework.anthropic` | Beta |
| Anthropic on Vertex | `agent-framework-anthropic` | `agent_framework.anthropic` | Beta |
| Claude Code SDK | `agent-framework-claude` | `agent_framework.anthropic` | Beta |
| Ollama | `agent-framework-ollama` | `agent_framework.ollama` | Beta |
| Amazon Bedrock (native) | `agent-framework-bedrock` | `agent_framework.amazon` | Beta |
| GitHub Copilot | `agent-framework-github-copilot` | `agent_framework.github` | Beta |
| Copilot Studio | `agent-framework-copilotstudio` | `agent_framework.microsoft` | Beta |

## OpenAI (and Azure OpenAI)

A single class — `OpenAIChatClient` — drives both OpenAI and Azure OpenAI. The routing is determined by which arguments you pass: `credential=` or `azure_endpoint=` select Azure; otherwise it stays on OpenAI.

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

# OpenAI — reads OPENAI_API_KEY and OPENAI_CHAT_MODEL from env
agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
)
response = await agent.run("Hello")
print(response.text)
```

Responses API vs Chat Completions API: `OpenAIChatClient` uses the **Responses API** (recommended — supports hosted tools like file search, code interpreter). `OpenAIChatCompletionClient` uses the classic Chat Completions API for OpenAI-compatible gateways that don't support `/responses`.

```python
from agent_framework.openai import OpenAIChatClient, OpenAIChatCompletionClient

responses_client = OpenAIChatClient(model="gpt-5")            # /responses
completions_client = OpenAIChatCompletionClient(model="gpt-5")  # /chat/completions
```

Azure OpenAI with Entra ID (passwordless):

```python
import os
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import AzureCliCredential

credential = AzureCliCredential()  # or DefaultAzureCredential()
agent = Agent(
    client=OpenAIChatClient(
        model=os.environ["AZURE_OPENAI_CHAT_MODEL"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        credential=credential,
    ),
    instructions="You are a helpful assistant.",
)
```

Azure OpenAI with API key:

```python
import os
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient(
    model=os.environ["AZURE_OPENAI_CHAT_MODEL"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)
```

Full-URL override (useful for reverse proxies): pass `base_url="https://…/openai/v1"` instead of `azure_endpoint=`.

Environment-variable cascade resolved inside the constructor:

| Argument | OpenAI env var | Azure env var |
|---|---|---|
| `model` | `OPENAI_CHAT_MODEL` → `OPENAI_MODEL` | `AZURE_OPENAI_CHAT_MODEL` → `AZURE_OPENAI_MODEL` |
| `api_key` | `OPENAI_API_KEY` | `AZURE_OPENAI_API_KEY` |
| `base_url` | `OPENAI_BASE_URL` | `AZURE_OPENAI_BASE_URL` |
| `azure_endpoint` | — | `AZURE_OPENAI_ENDPOINT` |
| `api_version` | — | `AZURE_OPENAI_API_VERSION` |
| `org_id` | `OPENAI_ORG_ID` | — |

## Microsoft Foundry

Microsoft Foundry (formerly Azure AI Foundry) provides project-scoped model deployments plus first-party evaluation and agent hosting. The client talks to the OpenAI-compatible endpoint surfaced by the Foundry project.

```python
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential

async with AzureCliCredential() as credential:
    agent = Agent(
        client=FoundryChatClient(
            project_endpoint="https://<project>.services.ai.azure.com",
            model="gpt-4o-mini",
            credential=credential,
        ),
        instructions="You are a helpful assistant.",
    )
    response = await agent.run("Summarise agent-framework 1.5.0 in one line.")
```

Env vars: `FOUNDRY_PROJECT_ENDPOINT`, `FOUNDRY_MODEL`.

If you already hold an `AIProjectClient`, pass it directly and skip endpoint/credential:

```python
from azure.ai.projects import AIProjectClient

project = AIProjectClient(endpoint=..., credential=...)
client = FoundryChatClient(project_client=project, model="gpt-4o-mini")
```

**Service-managed agents.** Use `FoundryAgent` when you want the agent's identity, threads, and tool definitions to live in Foundry (not in your process):

```python
from agent_framework.foundry import FoundryAgent

foundry_agent = FoundryAgent(
    project_endpoint="https://<project>.services.ai.azure.com",
    agent_name="contract-reviewer",
    agent_version="1.0",
    credential=credential,
)
response = await foundry_agent.run("Review contract.pdf")
```

## Foundry Local

`FoundryLocalClient` targets the local Foundry inference runtime (GGUF/ONNX models served by `foundry-local`). Useful for offline development and compliance scenarios.

```python
from agent_framework.foundry import FoundryLocalClient
from agent_framework import Agent

agent = Agent(
    client=FoundryLocalClient(model="Phi-3.5-mini-instruct"),
    instructions="You are a private offline assistant.",
)
```

## Anthropic

Three transports — direct Anthropic API, Anthropic on AWS Bedrock, Anthropic on Google Vertex. All three implement the same chat-client protocol, so only the construction differs.

```python
from agent_framework import Agent
from agent_framework.anthropic import (
    AnthropicClient,          # api.anthropic.com; reads ANTHROPIC_API_KEY
    AnthropicBedrockClient,   # Anthropic via AWS Bedrock
    AnthropicVertexClient,    # Anthropic via Google Vertex AI
)

agent = Agent(
    client=AnthropicClient(model="claude-sonnet-4-5"),
    instructions="You are a helpful assistant.",
)
```

Use the Claude Agent SDK instead of a chat client when you want Claude to drive its own tool loop, subagents, and session continuity:

```python
from agent_framework.anthropic import ClaudeAgent, ClaudeAgentOptions

claude = ClaudeAgent(
    options=ClaudeAgentOptions(model="claude-sonnet-4-5", permission_mode="default"),
)
response = await claude.run("Refactor utils.py to use dataclasses.")
```

## Ollama

Local models via the Ollama daemon.

```python
from agent_framework import Agent
from agent_framework.ollama import OllamaChatClient

agent = Agent(
    client=OllamaChatClient(model="llama3.1"),
    instructions="You are a helpful assistant.",
)
```

Custom base URL (non-default daemon):

```python
OllamaChatClient(model="llama3.1", base_url="http://gpu-host:11434")
```

## Amazon Bedrock (native)

The `agent_framework.amazon` namespace exposes the native Bedrock Converse API (for Titan, Nova, Mistral, Cohere, DeepSeek, etc. on Bedrock). For Claude on Bedrock, use `AnthropicBedrockClient` from the Anthropic provider instead — it unlocks Anthropic-specific features like extended thinking.

```python
from agent_framework import Agent
from agent_framework.amazon import BedrockChatClient

agent = Agent(
    client=BedrockChatClient(model="amazon.nova-pro-v1:0", region="us-east-1"),
    instructions="You are a helpful assistant.",
)
```

Guardrails:

```python
from agent_framework.amazon import BedrockChatClient, BedrockGuardrailConfig

client = BedrockChatClient(
    model="amazon.nova-pro-v1:0",
    guardrail=BedrockGuardrailConfig(guardrail_id="gr-xyz", guardrail_version="1"),
)
```

## GitHub Copilot

```python
from agent_framework import Agent
from agent_framework.github import CopilotChatClient  # agent_framework_github_copilot

agent = Agent(
    client=CopilotChatClient(model="gpt-4o"),
    instructions="Pair-programmer mode.",
)
```

## Copilot Studio

```python
from agent_framework.microsoft import CopilotStudioAgent  # agent_framework_copilotstudio

agent = CopilotStudioAgent(
    bot_id="<bot id>",
    tenant_id="<tenant id>",
    # …auth config…
)
```

## Swap providers at runtime

Because every client satisfies `SupportsChatGetResponse`, the agent stays identical — only the client changes:

```python
import os
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.anthropic import AnthropicClient
from agent_framework.ollama import OllamaChatClient

def build_client():
    provider = os.environ.get("LLM_PROVIDER", "openai")
    if provider == "anthropic":
        return AnthropicClient(model="claude-sonnet-4-5")
    if provider == "ollama":
        return OllamaChatClient(model="llama3.1")
    return OpenAIChatClient(model="gpt-5")

agent = Agent(client=build_client(), instructions="Helpful assistant.")
```

## Embeddings

Every provider with embedding support exposes an `*EmbeddingClient` alongside its chat client. All satisfy `SupportsGetEmbeddings` and return the same `GeneratedEmbeddings[list[float], EmbeddingGenerationOptions]` type, so you can swap them freely.

```python
from agent_framework.openai import OpenAIEmbeddingClient
from agent_framework.ollama import OllamaEmbeddingClient
from agent_framework.foundry import FoundryEmbeddingClient
from agent_framework.amazon import BedrockEmbeddingClient

embeddings = OpenAIEmbeddingClient(model="text-embedding-3-large")
result = await embeddings.get_embeddings(["hello", "world"])

for vec in result:
    print(vec.dimensions, vec.model, vec.vector[:4])
# `result.usage` is a UsageDetails (dict-like); key names vary by provider.
print("tokens used:", (result.usage or {}).get("total_tokens", 0))
```

### The `Embedding` and `GeneratedEmbeddings` types

`get_embeddings` always returns a `GeneratedEmbeddings` — it subclasses `list[Embedding]`, so iteration, indexing, and `len(...)` work as you'd expect. Each `Embedding` is generic over the vector type (usually `list[float]`, sometimes `list[int]` or `bytes` for quantised providers):

```python
from agent_framework import Embedding, GeneratedEmbeddings

# Constructing an Embedding directly — dimensions default to len(vector).
single = Embedding(vector=[0.1, 0.2, 0.3], model="text-embedding-3-small")
assert single.dimensions == 3

# Wrapping a list of them as a GeneratedEmbeddings — this is the shape your
# code should handle from every *EmbeddingClient.
batch = GeneratedEmbeddings(
    [single, Embedding(vector=[0.4, 0.5, 0.6])],
    usage={"prompt_tokens": 10, "total_tokens": 10},
)
assert len(batch) == 2
```

### Picking dimensions (OpenAI text-embedding-3-*)

The OpenAI `text-embedding-3-*` models support a `dimensions` parameter that lets you request a shorter vector without a separate model. Pass it through `OpenAIEmbeddingOptions`:

```python
from agent_framework.openai import OpenAIEmbeddingClient, OpenAIEmbeddingOptions

client = OpenAIEmbeddingClient(model="text-embedding-3-large")

# 256-dim embeddings — cheaper to store, 4x smaller vector DB footprint.
result = await client.get_embeddings(
    ["hello"],
    options=OpenAIEmbeddingOptions(dimensions=256, encoding_format="float"),
)
assert result[0].dimensions == 256
```

### Provider-neutral duck typing

Any code that embeds can take the `SupportsGetEmbeddings` protocol instead of a concrete class — type checkers will accept every first-party client and any subclass of `BaseEmbeddingClient` you write yourself:

```python
from agent_framework import SupportsGetEmbeddings


async def index(client: SupportsGetEmbeddings, docs: list[str]) -> list[list[float]]:
    result = await client.get_embeddings(docs)
    return [e.vector for e in result]
```

### Custom embedding client

Subclass `BaseEmbeddingClient` when you need to wrap a provider that isn't first-party or want to add batching/caching/shadowing on top of an existing one. The full pattern lives in the [Advanced page](./microsoft_agent_framework_python_advanced/#custom-embedding-client--baseembeddingclient); the short version:

```python
from agent_framework import BaseEmbeddingClient, Embedding, GeneratedEmbeddings


class StubEmbeddingClient(BaseEmbeddingClient):
    OTEL_PROVIDER_NAME = "stub"

    async def get_embeddings(self, values, *, options=None):
        return GeneratedEmbeddings(
            [Embedding(vector=[0.0] * 8, model="stub") for _ in values],
            options=options,
        )
```

## Provider-neutral request options — `ChatOptions`

Every provider-specific options TypedDict (`OpenAIChatOptions`, `AnthropicChatOptions`, etc.) extends the generic `ChatOptions` base. When you're writing code that should work against any client, type against `ChatOptions` — it captures the common denominator across all providers. All fields are optional (`total=False`), so you only set what you need.

```python
from agent_framework import ChatOptions

common: ChatOptions = {
    "model": "gpt-5-mini",
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 2_000,
    "stop": ["\n\nUSER:"],
    "seed": 1337,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.1,
    "user": "user-42",               # end-user id for provider-side abuse tracking
    "metadata": {"env": "prod"},     # attached to the request; provider may echo it back
}

response = await agent.run("Summarise the dataset.", options=common)
```

The fields `ChatOptions` defines — every first-party client accepts at least this subset:

| Field | Type | Purpose |
|---|---|---|
| `model` | `str` | Override the model for this one call |
| `temperature` / `top_p` | `float` | Sampling temperature and nucleus probability |
| `max_tokens` | `int` | Upper bound on output tokens |
| `stop` | `str \| Sequence[str]` | Stop sequences |
| `seed` | `int` | Reproducibility hint (providers may ignore) |
| `logit_bias` | `dict[str \| int, float]` | Per-token bias map |
| `frequency_penalty` / `presence_penalty` | `float` | Repetition / novelty penalties |
| `tools` | `Sequence[FunctionTool \| Callable \| …] \| None` | Per-call tool list (additive over the agent's) |
| `tool_choice` | `ToolMode \| "auto" \| "required" \| "none"` | Force a tool, require any tool, or disable tools |
| `allow_multiple_tool_calls` | `bool` | Permit the model to request more than one tool per turn |
| `response_format` | `type[BaseModel] \| Mapping \| None` | Structured output — pass a Pydantic class or a JSON schema |
| `metadata` | `dict[str, Any]` | Free-form metadata the provider round-trips on the request |
| `user` | `str` | End-user identifier (OpenAI / Anthropic use it for abuse detection) |
| `store` | `bool` | Provider-side conversation storage (OpenAI Responses API, Foundry) |
| `conversation_id` | `str` | Continue a provider-managed conversation |
| `instructions` | `str` | Per-call system instructions override |

### Forcing a specific tool with `ToolMode`

`tool_choice` accepts either the shorthand literal strings or a `ToolMode` TypedDict for fine-grained control over when and which tools the model may call. `ToolMode` has three optional keys:

| Key | Type | When to use |
|---|---|---|
| `mode` | `"auto"` \| `"required"` \| `"none"` | Required. `"auto"` lets the model decide; `"required"` forces at least one tool call; `"none"` disables all tools. |
| `required_function_name` | `str` | Only valid with `mode="required"`. Forces the model to call exactly this tool. |
| `allowed_tools` | `list[str]` | Valid with `"auto"` or `"required"`. Restricts which tools the model can see for this call. |

#### Force any tool — `mode="required"`

Use when you want to guarantee the model calls at least one tool (not just returns a text response):

```python
import asyncio
from typing import Annotated
from agent_framework import Agent, ChatOptions, ToolMode, tool
from agent_framework.openai import OpenAIChatClient


@tool
def get_weather(location: Annotated[str, "City name"]) -> str:
    """Return current weather for a city."""
    return f"The weather in {location} is 22°C."


@tool
def search_web(query: Annotated[str, "Search query"]) -> str:
    """Search the web for up-to-date information."""
    return f"Results for {query}: ..."


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a research assistant.",
    tools=[get_weather, search_web],
)


async def main() -> None:
    # Model MUST call a tool — useful when the pipeline expects a structured tool result
    response = await agent.run(
        "What's the weather in Amsterdam?",
        options=ChatOptions(tool_choice="required"),
    )
    print(response.text)


asyncio.run(main())
```

#### Pin to one specific tool — `required_function_name`

Use when you know exactly which tool should run. The model fills in the arguments:

```python
import asyncio
from agent_framework import Agent, ChatOptions, ToolMode, tool
from agent_framework.openai import OpenAIChatClient


@tool
def lookup_customer(customer_id: str) -> str:
    """Look up a customer by ID."""
    return f"Customer {customer_id}: Alice Smith"


@tool
def cancel_subscription(customer_id: str, reason: str) -> str:
    """Cancel a customer's subscription."""
    return f"Subscription for {customer_id} cancelled: {reason}"


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a billing assistant.",
    tools=[lookup_customer, cancel_subscription],
)


async def main() -> None:
    # Force lookup_customer — the model must call this tool first regardless of what it thinks
    response = await agent.run(
        "Customer c-4821 wants to cancel.",
        options=ChatOptions(
            tool_choice=ToolMode(mode="required", required_function_name="lookup_customer")
        ),
    )
    print(response.text)


asyncio.run(main())
```

#### Restrict visible tools — `allowed_tools`

Use when only a subset of the agent's tools is relevant for a particular turn. The model can only see and call the listed tools:

```python
import asyncio
from agent_framework import Agent, ChatOptions, ToolMode, tool
from agent_framework.openai import OpenAIChatClient


@tool
def search_products(query: str) -> str:
    """Search the product catalog."""
    return f"Found 12 results for {query}"


@tool
def add_to_cart(product_id: str, qty: int) -> str:
    """Add a product to the shopping cart."""
    return f"Added {qty}x {product_id} to cart"


@tool
def checkout(cart_id: str) -> str:
    """Complete the checkout flow."""
    return f"Order placed for cart {cart_id}"


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a shopping assistant.",
    tools=[search_products, add_to_cart, checkout],
)


async def main() -> None:
    # Discovery phase — only search is allowed
    await agent.run(
        "Find red sneakers under $100.",
        options=ChatOptions(
            tool_choice=ToolMode(mode="auto", allowed_tools=["search_products"])
        ),
    )

    # Add-to-cart phase — search is still available but checkout is blocked
    await agent.run(
        "Add product SKU-4821 to my cart.",
        options=ChatOptions(
            tool_choice=ToolMode(mode="required", allowed_tools=["add_to_cart"])
        ),
    )

    # Checkout phase — only checkout is visible
    await agent.run(
        "Place the order.",
        options=ChatOptions(
            tool_choice=ToolMode(mode="required", allowed_tools=["checkout"])
        ),
    )


asyncio.run(main())
```

#### Disable tools entirely — `mode="none"`

Use when you want a plain text response without any tool calls — for example a final summary step after all research is done:

```python
import asyncio
from agent_framework import Agent, ChatOptions, tool
from agent_framework.openai import OpenAIChatClient


@tool
def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for {query}: ..."


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a research assistant.",
    tools=[search_web],
)


async def main() -> None:
    session = agent.create_session()

    # Research phase — tools enabled
    await agent.run("Research the history of quantum computing.", session=session)

    # Summary phase — force plain text, no tool calls
    summary = await agent.run(
        "Now give me a concise two-paragraph summary of what you found.",
        session=session,
        options=ChatOptions(tool_choice="none"),
    )
    print(summary.text)


asyncio.run(main())
```

`mode="none"` disables tools entirely for one call (useful when you want a pure summary of the conversation without further tool-use); `mode="required"` without `required_function_name` forces the model to pick *some* tool.

### Structured output in one line

`response_format` accepts a Pydantic model — the response comes back as a typed object via `response.value`:

```python
from pydantic import BaseModel
from agent_framework import ChatOptions

class Extracted(BaseModel):
    sentiment: str
    score: float
    topics: list[str]

options: ChatOptions = {"response_format": Extracted}
response = await agent.run(
    "Summarise this review: 'Fast shipping, but the fabric snagged.'",
    options=options,
)
print(response.value.sentiment, response.value.score)
```

Providers that don't support structured output natively fall back to JSON-mode + client-side validation — same surface either way.

### Merging with client-level defaults

Every client accepts the same TypedDict on construction. The call-level `options=` is a shallow merge on top: keys you set win, keys you omit inherit from the client.

```python
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient(model="gpt-5-mini", temperature=0.7)

# Inherits temperature=0.7; only max_tokens is overridden.
await agent.run("Draft a tweet.", options={"max_tokens": 280})
```

## Swapping the model on an existing client

Every client accepts `model=` at construction and remembers it. But you can also override the model for a single call without building a new client — use `options=` on the agent run:

```python
from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions

default_client = OpenAIChatClient(model="gpt-5-mini")

agent = Agent(client=default_client, instructions="…")

# Upgrade to a bigger model just for this one tricky question.
response = await agent.run(
    "Prove Fermat's last theorem in two sentences.",
    options=OpenAIChatOptions(model="gpt-5", temperature=0.2),
)
```

`options` is a provider-specific `TypedDict` — `OpenAIChatOptions`, `OpenAIChatCompletionOptions`, `OpenAIEmbeddingOptions`, and the equivalents under `agent_framework.anthropic`, `agent_framework.amazon`, `agent_framework.ollama`, etc. IDE autocomplete drives you through every tunable. The values merge with the client's defaults; anything you omit stays as the client was constructed.

### When to reach for the provider-specific TypedDict

Use the generic `ChatOptions` whenever the knobs you need are common across providers — that keeps the call site interoperable. Drop to the provider-specific dict only when you need a feature the base can't describe:

- **OpenAI-only** (via `OpenAIChatOptions`): `reasoning`, `prompt_cache_key`, `prompt_cache_retention`, `service_tier`, `top_logprobs`, `truncation`, `background`, `include`, `max_tool_calls`, `continuation_token`.
- **Anthropic-only**: extended-thinking parameters, cache-control directives.
- **Bedrock-only**: `guardrail` references, `additional_model_request_fields`.

Mixing them is fine — a provider-specific dict is a superset of `ChatOptions`, so code typed against the base still accepts it.

## Building your own chat client

For a provider that isn't in the first-party list, or to wrap an existing client with caching / shadow traffic / logging, subclass `BaseChatClient`. Implement one method — `_inner_get_response` — and inherit middleware, telemetry, and the function calling loop for free. See the full recipe in [Advanced → Custom chat client](./microsoft_agent_framework_python_advanced/#custom-chat-client--basechatclient).

## Picking a provider

- **Prototyping** — `OpenAIChatClient()` or `OllamaChatClient(model="llama3.1")`. Neither requires Azure tooling.
- **Azure-native deployments** — `OpenAIChatClient` with `azure_endpoint` + `credential`, or `FoundryChatClient` if you're already on a Foundry project (evaluation, service-managed agents, private networking).
- **Cross-cloud Claude** — `AnthropicClient` for Anthropic direct; `AnthropicBedrockClient` or `AnthropicVertexClient` to keep data in AWS/GCP.
- **Offline / compliance** — `OllamaChatClient` or `FoundryLocalClient`.
- **Multi-provider fallback** — build a thin factory (example above) and let an env var pick at startup; the rest of your agent code stays unchanged.

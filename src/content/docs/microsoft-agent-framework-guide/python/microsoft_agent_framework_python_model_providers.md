---
title: "Microsoft Agent Framework (Python) — Model Providers"
description: "Real imports and constructors for every first-party chat client in agent-framework 1.1.0: OpenAI, Azure OpenAI, Microsoft Foundry, Foundry Local, Anthropic, Ollama, Bedrock, GitHub Copilot, Copilot Studio."
framework: microsoft-agent-framework
language: python
---

# Model Providers — Python

Every chat client in `agent-framework` implements the same `SupportsChatGetResponse` protocol, so `Agent(client=...)` accepts them interchangeably. The import is always `agent_framework.<provider>.<ClassName>` — **no Azure SDK import is required for any of these**. The Azure SDK only becomes relevant for authentication (`azure-identity`) or for Azure-specific storage providers.

This page was verified against `agent-framework-core==1.1.0` and provider packages at `1.0.0b260421` (April 2026). Each sub-package is imported lazily from the `agent_framework.<provider>` namespace — you install the provider package and import from `agent_framework.<provider>`.

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
    response = await agent.run("Summarise agent-framework 1.1.0 in one line.")
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

Every provider with embedding support exposes an `*EmbeddingClient` alongside its chat client:

```python
from agent_framework.openai import OpenAIEmbeddingClient
from agent_framework.ollama import OllamaEmbeddingClient
from agent_framework.foundry import FoundryEmbeddingClient
from agent_framework.amazon import BedrockEmbeddingClient

embeddings = OpenAIEmbeddingClient(model="text-embedding-3-large")
vectors = await embeddings.get_embeddings(["hello", "world"])
```

## Picking a provider

- **Prototyping** — `OpenAIChatClient()` or `OllamaChatClient(model="llama3.1")`. Neither requires Azure tooling.
- **Azure-native deployments** — `OpenAIChatClient` with `azure_endpoint` + `credential`, or `FoundryChatClient` if you're already on a Foundry project (evaluation, service-managed agents, private networking).
- **Cross-cloud Claude** — `AnthropicClient` for Anthropic direct; `AnthropicBedrockClient` or `AnthropicVertexClient` to keep data in AWS/GCP.
- **Offline / compliance** — `OllamaChatClient` or `FoundryLocalClient`.
- **Multi-provider fallback** — build a thin factory (example above) and let an env var pick at startup; the rest of your agent code stays unchanged.

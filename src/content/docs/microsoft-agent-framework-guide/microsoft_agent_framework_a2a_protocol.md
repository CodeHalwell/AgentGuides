---
title: "Microsoft Agent Framework — A2A Protocol"
description: "The real A2AAgent in agent-framework-a2a — connect to any A2A-compliant external agent via HTTP/JSON-RPC."
framework: microsoft-agent-framework
---

# A2A Protocol

> **Errata (April 2026).** An earlier draft of this page described a sprawling `agent_framework.a2a` submodule with classes like `A2AProtocolAdapter`, `A2AClient`, `A2AMessage`, `OAuth2Authentication`, `EntraIdAuthentication`, `MutualTLSAuthentication`, `ApiKeyAuthentication`, `AccessPolicy`, `SigningConfig`, `RateLimitConfig`, `CircuitBreakerConfig`, and `EncryptionConfig`. **None of those classes exist.** The `agent_framework.a2a` package exports exactly one class: `A2AAgent`. This page was rewritten after direct introspection of `agent-framework-a2a==1.0.0b260428`.

## What A2A actually is

[Agent2Agent (A2A)](https://github.com/google/A2A) is an open protocol for agents built on different frameworks to talk to each other over HTTP/JSON-RPC. The Agent Framework integration is a thin wrapper: `A2AAgent` makes any A2A-compliant remote agent look like a native framework `Agent`.

All the security, auth, transport, and rate-limiting concerns the old draft invented as bespoke classes are actually handled by:

- **HTTP client** — you pass an `httpx.AsyncClient` with whatever TLS / auth / timeout config you need.
- **A2A auth** — the underlying [`a2a-sdk`](https://pypi.org/project/a2a-sdk/) package provides `AuthInterceptor` which you configure yourself and hand in.
- **Upstream protocol behaviours** — retries, circuit breakers, encryption, message signing — are the A2A **protocol's** concerns, not framework-specific classes. Use the `a2a-sdk` docs.

## Install

```bash
pip install agent-framework-a2a --pre
```

This pulls in the meta package dependencies and `a2a-sdk`.

## Verified signature (April 2026)

```python
A2AAgent(
    *,
    name: str | None = None,
    id: str | None = None,
    description: str | None = None,
    agent_card: AgentCard | None = None,
    url: str | None = None,
    client: a2a.client.Client | None = None,
    http_client: httpx.AsyncClient | None = None,
    auth_interceptor: AuthInterceptor | None = None,
    timeout: float | httpx.Timeout | None = None,
    **kwargs: Any,
)
```

Key methods (verified):
- `.run(query)` — standard `Agent.run` surface; returns an `AgentResponse`.
- `.poll_task(...)` — poll a long-running A2A task.
- `.as_tool()` — expose the remote agent as a local tool.
- `.create_session()` / `.get_session()` — session management like any other agent.

## Minimum viable client

Connect to an A2A-hosted agent by URL:

```python
import asyncio
from agent_framework.a2a import A2AAgent

async def main():
    remote = A2AAgent(
        name="RemoteAnalyst",
        url="https://analyst.example.com/a2a",
    )
    response = await remote.run("Summarise Q3 sales")
    print(response.text)

asyncio.run(main())
```

Connect using a discovered `AgentCard`:

```python
from agent_framework.a2a import A2AAgent
from a2a.types import AgentCard

card = AgentCard.from_url("https://analyst.example.com/a2a/.well-known/agent.json")

remote = A2AAgent(name="RemoteAnalyst", agent_card=card)
```

## Authentication

Pass an `AuthInterceptor` from `a2a-sdk`. The framework itself does not define auth classes; use the A2A SDK's:

```python
from agent_framework.a2a import A2AAgent
from a2a.client import AuthInterceptor, BearerTokenCredentials

remote = A2AAgent(
    url="https://analyst.example.com/a2a",
    auth_interceptor=AuthInterceptor(
        credentials=BearerTokenCredentials(token="..."),
    ),
)
```

For production-grade auth (OAuth2 / Entra ID / mTLS / API keys), configure them at the `httpx.AsyncClient` transport level or via `a2a-sdk`'s auth plumbing — the Agent Framework just passes your client through.

## Exposing a local agent over A2A

Serving a framework `Agent` *as* an A2A endpoint is out of scope for the `agent-framework-a2a` package. Use the [Azure AI Agent Server](https://pypi.org/project/azure-ai-agentserver-agentframework/) or host your own A2A server per the [A2A specification](https://github.com/google/A2A). The framework provides `agent.run` / `agent.as_tool()`; the HTTP surface is your server's responsibility.

## What was removed from this page

All of the following appeared in earlier drafts and are not real:

- `from agent_framework.a2a import A2AProtocolAdapter` — no such class.
- `from agent_framework.a2a import A2AClient, A2AMessage` — not in this module (the real A2A client is `a2a.client.Client` from the `a2a-sdk` package).
- `OAuth2Authentication`, `EntraIdAuthentication`, `MutualTLSAuthentication`, `ApiKeyAuthentication` — not framework classes. Auth is handled via `httpx` / `a2a-sdk`'s `AuthInterceptor`.
- `AccessPolicy`, `SigningConfig`, `RateLimitConfig`, `CircuitBreakerConfig`, `EncryptionConfig` — not real classes. Those concerns live in the transport layer (your `httpx.AsyncClient` or API gateway).
- Framework-defined cross-framework "protocol adapters" for OpenAI SDK / Claude SDK / LangGraph / Google ADK — A2A itself is the protocol; no framework-specific adapter is needed because each participant just implements A2A.

## Further reading

- [A2A protocol spec](https://github.com/google/A2A)
- [`a2a-sdk` on PyPI](https://pypi.org/project/a2a-sdk/)
- Source: <https://github.com/microsoft/agent-framework>

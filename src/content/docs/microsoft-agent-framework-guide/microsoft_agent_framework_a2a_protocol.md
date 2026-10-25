---
title: "Microsoft Agent Framework — A2A Protocol"
description: "Call remote Agent2Agent (A2A) agents with A2AAgent and serve your own agents over A2A with A2AExecutor, in Python and .NET."
framework: "microsoft-agent-framework"
sidebar:
  label: "A2A protocol"
  order: 20
---

[Agent2Agent (A2A)](https://a2a-protocol.org/) is an open protocol that lets agents built on different frameworks call each other over HTTP (JSON-RPC, HTTP+JSON or gRPC bindings). Microsoft Agent Framework supports both directions:

- **Client:** a remote A2A agent becomes a local agent (`A2AAgent` in Python, an `AIAgent` from `A2ACardResolver` in .NET), so it can be run, streamed, placed in a workflow or exposed as a tool.
- **Server:** any framework agent (or a workflow wrapped with `as_agent()`) can be published as an A2A endpoint (`A2AExecutor` in Python, `MapA2AJsonRpc` in .NET).

Protocol-level concerns (the wire format, task lifecycle, agent cards, auth schemes) come from the A2A SDKs. The framework adds the adapter between its agent abstraction and those SDKs.

## Python

```bash
pip install agent-framework-a2a --pre    # pulls in a2a-sdk
```

`agent_framework.a2a` exports four names:

| Name | Purpose |
|---|---|
| `A2AAgent` | Client: wraps a remote A2A agent as a local agent. |
| `A2AAgentSession` | Session type for `A2AAgent`; carries the A2A `context_id`, `task_id` and `task_state` between turns. |
| `A2AExecutor` | Server: an `a2a-sdk` `AgentExecutor` that runs a framework `Agent` or `WorkflowAgent`. |
| `A2AServiceSessionId` | The service-side session id `A2AAgentSession` stores. |

### Serving an agent with `A2AExecutor`

`A2AExecutor(agent, stream=False, run_kwargs=None)` plugs into the `a2a-sdk` server stack: a `DefaultRequestHandler` with a task store, plus Starlette routes for the agent card and the JSON-RPC endpoint. With `stream=True` the agent's updates are sent as A2A artifact updates as they arrive. `run_kwargs` is forwarded to every `agent.run(...)` call (for example `{"client_kwargs": {...}}`).

```python
import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from starlette.applications import Starlette

from agent_framework import Agent
from agent_framework.a2a import A2AExecutor
from agent_framework.openai import OpenAIChatClient

agent = Agent(
    client=OpenAIChatClient(),
    name="analyst",
    instructions="You summarise sales figures.",
)

card = AgentCard(
    name="Sales analyst",
    description="Summarises sales figures.",
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=True),
    supported_interfaces=[
        AgentInterface(url="http://localhost:9999/", protocol_binding="JSONRPC", protocol_version="1.0"),
    ],
    skills=[AgentSkill(id="summarise", name="Summarise", description="Summarise sales data.", tags=["sales"])],
)

handler = DefaultRequestHandler(
    agent_executor=A2AExecutor(agent, stream=True),
    task_store=InMemoryTaskStore(),
    agent_card=card,
)

app = Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9999)
```

The card is served at `/.well-known/agent-card.json`. Points to settle before production:

- **Task store.** `InMemoryTaskStore` loses tasks on restart and is per-process; `a2a-sdk` ships database-backed task stores for multi-instance deployments.
- **Authentication.** Declare the scheme in the card (`security_schemes` / `security_requirements`) and enforce it in Starlette middleware or at your gateway. The executor does not authenticate callers.
- **History.** For each task the executor creates a fresh session whose `session_id` is the A2A context id. The default in-memory history lives on that session object and is discarded after the task, so give the agent a history provider keyed by `session_id` (`FileHistoryProvider`, or the Redis and Cosmos DB providers) if callers hold multi-turn conversations.

To publish a workflow instead of a single agent, pass `workflow.as_agent(name=...)` to `A2AExecutor`.

### Calling a remote agent with `A2AAgent`

```python
import asyncio
from agent_framework.a2a import A2AAgent


async def main() -> None:
    remote = A2AAgent(name="analyst", url="http://localhost:9999/")
    response = await remote.run("Summarise Q3 sales")
    print(response.text)


asyncio.run(main())
```

`A2AAgent` takes either `url=` (the card is resolved for you) or a pre-fetched `agent_card=`. Other keyword arguments: `http_client=` (your own `httpx.AsyncClient`), `auth_interceptor=`, `timeout=`, `supported_protocol_bindings=` (`"JSONRPC"`, `"GRPC"`, `"HTTP+JSON"`) and `forwarded_kwargs=` (names of keys from the `function_invocation_kwargs` / `client_kwargs` you pass to `run()` that should be sent to the server in the request metadata).

To inspect the card first, for example to check its skills, use `A2ACardResolver` from `a2a-sdk`:

```python
import asyncio
import httpx
from a2a.client import A2ACardResolver
from agent_framework.a2a import A2AAgent


async def main() -> None:
    async with httpx.AsyncClient(timeout=30) as http:
        resolver = A2ACardResolver(httpx_client=http, base_url="http://localhost:9999")
        card = await resolver.get_agent_card()        # GET /.well-known/agent-card.json
        print(card.name, [s.name for s in card.skills])

        remote = A2AAgent(name=card.name, agent_card=card, http_client=http)
        print((await remote.run("Hello")).text)


asyncio.run(main())
```

A remote agent behaves like any other agent: keep a session (`remote.create_session()`) for multi-turn conversations, use `stream=True`, add it to a `WorkflowBuilder`, or give it to another agent with `remote.as_tool()`. When the remote task ends in `input-required`, the next `run()` on the same session continues that task. Long-running tasks surface an `A2AContinuationToken` on `response.continuation_token`; pass it back with `remote.run(None, session=session, continuation_token=token)` until it is `None`.

### Authentication

`AuthInterceptor` (from `a2a-sdk`) reads the security schemes the remote agent's card declares and asks a `CredentialService` for each credential. It adds a bearer token for HTTP-bearer, OAuth2 and OpenID Connect schemes, and an API-key header for API-key schemes:

```python
import os
from a2a.client import AuthInterceptor, CredentialService
from agent_framework.a2a import A2AAgent


class StaticToken(CredentialService):
    """Returns one token for every scheme the agent card declares."""

    def __init__(self, token: str) -> None:
        self._token = token

    async def get_credentials(self, security_scheme_name, context):
        return self._token


remote = A2AAgent(
    name="analyst",
    url="https://analyst.example.com/a2a/",
    auth_interceptor=AuthInterceptor(credential_service=StaticToken(os.environ["ANALYST_TOKEN"])),
)
```

For per-user tokens, `a2a-sdk`'s `InMemoryContextCredentialStore` keys credentials by session. For mTLS, or headers the card doesn't declare, configure them on the `httpx.AsyncClient` you pass as `http_client=`.

## .NET

```bash
dotnet add package Microsoft.Agents.AI.A2A --prerelease
dotnet add package Microsoft.Agents.AI.Hosting.A2A.AspNetCore --prerelease
```

`A2ACardResolver.GetAIAgentAsync()` turns a remote card into an `AIAgent`, and `builder.AddA2AServer(agent)` plus `app.MapA2AJsonRpc(agent, "/path")` (or `MapA2AHttpJson`) publish a local agent. Worked examples are in the [.NET feature tour](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_2025_features/#agent2agent-a2a-protocol---net).

## Further reading

- [A2A specification](https://a2a-protocol.org/latest/specification/)
- [`a2a-sdk` on PyPI](https://pypi.org/project/a2a-sdk/)
- [Microsoft Agent Framework source](https://github.com/microsoft/agent-framework) (`python/packages/a2a`)

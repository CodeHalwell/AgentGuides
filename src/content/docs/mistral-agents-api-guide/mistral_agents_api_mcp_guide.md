---
title: "Mistral Agents API: Model Context Protocol (MCP) Guide"
description: "Using MCP servers with Mistral agents: registered connectors, client-side MCP with RunContext, building servers with the mcp 2.x SDK, and human approval for tool calls."
framework: "mistral-agents-api"
language: "python"
sidebar:
  label: "MCP"
  order: 4
---

**Verified against `mistralai` %%version=mistralai%%** (checked %%versions-checked%%), with `mcp` 2.x as installed by `mistralai[agents]`.

The [Model Context Protocol](https://modelcontextprotocol.io) is an open standard for exposing tools and data to models. Mistral supports it in two places, and choosing between them is the main design decision:

| | Registered connector | Client-side MCP (`RunContext`) |
|---|---|---|
| Who calls the MCP server | Mistral's platform | your Python process |
| Server must be reachable from | the internet | your process (stdio, localhost, private network) |
| Transport | remote (streamable HTTP) | stdio, SSE or streamable HTTP |
| Credentials | stored with Mistral (`create_credentials`, OAuth) | stay in your process |
| Works with stored agents in any client | yes | only where your code runs the loop |
| SDK surface | `client.beta.connectors`, `{"type": "connector"}` tool | `mistralai.extra.run.context.RunContext`, `mistralai.extra.mcp.*` |

There is no `{"type": "mcp"}` agent tool; remote MCP servers are attached as connectors. Registered connectors are a public preview, and the Agents and Conversations APIs they plug into are beta.

```bash
pip install "mistralai[agents]==%%version=mistralai%%"   # RunContext + mcp>=2.2
```

---

## 1. Registered connectors

Register a remote MCP server once; then any agent or conversation can use it by name.

```python
import os
from mistralai.client import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

client.beta.connectors.create(
    request={
        "protocol": "mcp",
        "name": "tickets",
        "description": "Internal ticketing system: search, read and comment on tickets",
        "server": "https://mcp.tickets.example.com/mcp",
        "visibility": "private",
    }
)

# Store credentials that Mistral will send to the server
client.beta.connectors.create_credentials(
    connector_id_or_name="tickets",
    consumer_scope="workspace",
    name="default",
    credentials={"headers": {"Authorization": f"Bearer {os.environ['TICKETS_TOKEN']}"}},
    is_default=True,
)

for tool in client.beta.connectors.list_tools(connector_id_or_name="tickets"):
    print(tool.name if hasattr(tool, "name") else tool["name"])
```

Attach it to an agent, optionally limiting which tools are exposed:

```python
triager = client.beta.agents.create(
    model="mistral-medium-latest",
    name="ticket-triager",
    description="Finds and summarises related tickets.",
    instructions="Search tickets before answering. Never post comments without being asked.",
    tools=[{
        "type": "connector",
        "connector_id": "tickets",
        "tool_configuration": {"include": ["search_tickets", "get_ticket"]},
    }],
)

response = client.beta.conversations.start(
    agent_id=triager.id,
    inputs="Are there open tickets about login failures on Android?",
)
```

Servers using OAuth get an authorisation URL to send the user to: `client.beta.connectors.get_auth_url(connector_id_or_name=..., app_return_url=...)`. Since SDK v3, credential deletion is a single call, `delete_credentials(..., consumer_scope="user" | "workspace" | "organization")`.

---

## 2. Client-side MCP with `RunContext`

`RunContext` connects to MCP servers from your own process, advertises their tools to the model, executes the calls the model makes, and loops until it has an answer.

### Local stdio server

```python
import asyncio
import os
from mcp import StdioServerParameters
from mistralai.client import Mistral
from mistralai.extra.mcp.stdio import MCPClientSTDIO
from mistralai.extra.run.context import RunContext

async def main() -> None:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    filesystem = MCPClientSTDIO(
        stdio_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/srv/reports"],
        ),
        name="filesystem",
    )

    async with RunContext(model="mistral-medium-latest") as run_ctx:
        await run_ctx.register_mcp_client(
            mcp_client=filesystem,
            tool_configuration={"include": ["list_directory", "read_text_file"]},
        )
        result = await client.beta.conversations.run_async(
            run_ctx=run_ctx,
            inputs="Which report in /srv/reports mentions Q3 churn, and what does it conclude?",
            instructions="Use the filesystem tools. Quote the passage you rely on.",
        )
        print(result.output_as_text)

asyncio.run(main())
```

### Remote server over streamable HTTP or SSE

```python
import os
from mistralai.extra.mcp.sse import MCPClientSSE, SSEServerParams
from mistralai.extra.mcp.streamable_http import MCPClientStreamableHTTP, StreamableHTTPServerParams

inventory = MCPClientStreamableHTTP(
    params=StreamableHTTPServerParams(
        url="https://inventory.internal.example.com/mcp",
        headers={"Authorization": f"Bearer {os.environ['INVENTORY_TOKEN']}"},
        timeout=30,
    ),
    name="inventory",
)

legacy = MCPClientSSE(sse_params=SSEServerParams(url="https://legacy.internal.example.com/sse"), name="legacy")
```

Register several at once with `await run_ctx.register_mcp_clients([inventory, legacy])`. `MCPClientSSE` also accepts `oauth_params` for servers that implement MCP's OAuth flow.

`StreamableHTTPServerParams` does not follow redirects by default, so the `headers` you set are never forwarded to another host; leave `follow_redirects=False` unless every redirect target is trusted.

### Mixing MCP, your functions and built-in tools

```python
import asyncio
import os
from mistralai.client import Mistral
from mistralai.extra.run.context import RunContext

def convert_currency(amount: float, rate: float) -> float:
    """Convert an amount using an exchange rate.

    Args:
        amount: The amount to convert.
        rate: Units of the target currency per unit of the source currency.
    """
    return round(amount * rate, 2)

async def main() -> None:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    async with RunContext(model="mistral-medium-latest") as run_ctx:
        await run_ctx.register_mcp_client(mcp_client=inventory)   # from the previous example
        run_ctx.register_func(convert_currency)
        result = await client.beta.conversations.run_async(
            run_ctx=run_ctx,
            inputs="What is our stock value of SKU 4471 in euros at today's rate?",
            tools=[{"type": "web_search"}],     # server-side tools still work alongside
        )
        print(result.output_as_text)

asyncio.run(main())
```

To run against a stored agent, use `RunContext(agent_id=agent.id)` instead of `model=`. To continue a conversation, set `run_ctx.conversation_id` before calling `run_async()`. `run_stream_async()` yields events as the run progresses.

---

## 3. Human approval for tool calls

Tools that change things (send email, post comments, issue refunds) should wait for a person. Mark them as requiring confirmation; the run then stops with a `DeferredToolCallsException` that you can serialise, show to a user, and resume.

```python
import asyncio
import json
import os
from mistralai.client import Mistral
from mistralai.extra.exceptions import DeferredToolCallsException
from mistralai.extra.run.context import RunContext

def issue_refund(order_id: str, amount: float) -> str:
    """Issue a refund for an order.

    Args:
        order_id: The order to refund.
        amount: Amount in euros.
    """
    return f"Refunded {amount:.2f} EUR on {order_id}"

async def first_turn(client: Mistral) -> dict | None:
    async with RunContext(model="mistral-medium-latest") as run_ctx:
        run_ctx.register_func(issue_refund, requires_confirmation=True)
        try:
            result = await client.beta.conversations.run_async(
                run_ctx=run_ctx,
                inputs="Order ORD-881 arrived broken. Please refund the 49.90 EUR.",
            )
            print(result.output_as_text)
            return None
        except DeferredToolCallsException as deferred:
            for call in deferred.deferred_calls:
                print("needs approval:", call.tool_name, call.arguments)
            return deferred.to_dict()          # store in your database or send to a UI

async def resume(client: Mistral, state: dict, approved: bool) -> None:
    deferred = DeferredToolCallsException.from_dict(state)
    inputs = list(deferred.executed_results)
    for call in deferred.deferred_calls:
        inputs.append(call.confirm() if approved else call.reject("Refund declined by a supervisor"))

    async with RunContext(model="mistral-medium-latest") as run_ctx:
        run_ctx.conversation_id = deferred.conversation_id
        run_ctx.register_func(issue_refund, requires_confirmation=True)
        result = await client.beta.conversations.run_async(run_ctx=run_ctx, inputs=inputs)
        print(result.output_as_text)

async def main() -> None:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    state = await first_turn(client)
    if state:
        print(json.dumps(state)[:200])
        await resume(client, state, approved=True)

asyncio.run(main())
```

For client-side MCP tools, pass `tool_configuration={"requires_confirmation": ["post_comment"]}` to `register_mcp_client()`; the run then pauses in the same way.

### Approving connector and built-in tool calls

Registered connectors and built-in tools run on Mistral's side, so the approval goes through the conversation itself. List the tool names in the tool's `tool_configuration.requires_confirmation`; Mistral then stops before running them and returns a `function.call` entry with `confirmation_status == "pending"`. Answer with `tool_confirmations` on the next `append()`:

```python
from mistralai.client.models import FunctionCallEntry

response = client.beta.conversations.start(
    model="mistral-medium-latest",
    inputs="Add a comment to ticket T-42 saying the fix is deployed.",
    tools=[{
        "type": "connector",
        "connector_id": "tickets",
        "tool_configuration": {"requires_confirmation": ["post_comment"]},
    }],
)

pending = [
    e for e in response.outputs
    if isinstance(e, FunctionCallEntry) and e.confirmation_status == "pending"
]
if pending:
    confirmations = []
    for call in pending:
        print("needs approval:", call.name, call.arguments)
        approved = True                                   # ask a person here
        confirmations.append({"tool_call_id": call.tool_call_id, "confirmation": "allow" if approved else "deny"})
    response = client.beta.conversations.append(
        conversation_id=response.conversation_id,
        tool_confirmations=confirmations,
    )
print(response.outputs[-1].content)
```

`"deny"` tells the model the call was refused, and it answers accordingly. Because the conversation is stored, the approval can come in a later request: persist the `conversation_id` and the pending `tool_call_id`s. The same works for built-in tools, for example `{"type": "web_search_premium", "tool_configuration": {"requires_confirmation": ["web_search", "news_search"]}}`.

---

## 4. Building an MCP server

Any MCP server works with both routes. In Python, the `mcp` SDK (version 2.x, as installed with `mistralai[agents]`) provides `MCPServer`, which was called `FastMCP` in `mcp` 1.x:

```python
# tickets_server.py
from mcp.server.mcpserver import MCPServer

server = MCPServer(name="tickets", instructions="Search and read support tickets.")

TICKETS = {
    "T-101": {"title": "Login fails on Android 15", "status": "open"},
    "T-102": {"title": "Invoice PDF missing VAT number", "status": "closed"},
}

@server.tool()
def search_tickets(query: str, status: str = "open") -> list[dict]:
    """Search tickets by text and status."""
    q = query.lower()
    return [
        {"id": tid, **t} for tid, t in TICKETS.items()
        if q in t["title"].lower() and t["status"] == status
    ]

@server.tool()
def get_ticket(ticket_id: str) -> dict:
    """Return one ticket by ID."""
    return {"id": ticket_id, **TICKETS[ticket_id]}

if __name__ == "__main__":
    server.run()      # stdio; use transport="streamable-http" to serve it over HTTP as a connector
```

Run it locally with `MCPClientSTDIO(stdio_params=StdioServerParameters(command="python", args=["tickets_server.py"]))`. To use it as a registered connector instead, switch to `server.run(transport="streamable-http")` (it serves `/mcp` on port 8000), deploy it behind HTTPS and register that URL. A stdio client cannot talk to the HTTP transport, and the other way round.

Design advice for tools:

- **Small, well-named tools** with clear docstrings beat one generic `run_query` tool; the model chooses tools from their names and descriptions.
- **Typed parameters** with defaults; return structured data (dicts, lists) rather than prose.
- **Read and write separated**, so write tools can require confirmation.
- **Bounded results**: page or truncate large results; the model pays for every token.
- **Errors as results**: raise or return clear messages ("ticket T-999 not found") so the model can recover.

---

## 5. Security

- **Authenticate the server.** Remote MCP servers should require a bearer token or OAuth; a connector stores the credentials, a client-side MCP client sends them as headers.
- **Least privilege.** Give the server's own credentials (database users, API keys) only the permissions its tools need, and expose only the tools an agent needs with `include`.
- **Treat tool output as untrusted.** Documents, tickets and web pages can contain prompt injection. Require confirmation for anything that writes, sends or spends.
- **Keep secrets out of prompts.** Credentials belong in connector credentials or client headers, never in `instructions` or `inputs`.
- **Log tool calls.** `result.entries` (from `run_async`) and `response.outputs` (from `start`) contain every tool call and its arguments.
- **Pin versions.** `mcp` 2.x changed result field names (`structured_content` rather than `structuredContent`) and renamed `FastMCP`; pin `mistralai` and `mcp` together.

---

## 6. Production notes

- Client-side MCP servers started over stdio live as long as the `RunContext`. For a web service, create the context per request, or keep long-lived remote servers and connect over HTTP.
- Give remote servers health checks and timeouts; `StreamableHTTPServerParams(timeout=..., sse_read_timeout=...)` bound how long a call may take.
- Rate-limit expensive tools on the server side, since the model may call a tool several times in one run.
- Containerise servers like any other web service, and run them on the same network as the data they expose rather than on the agent host.

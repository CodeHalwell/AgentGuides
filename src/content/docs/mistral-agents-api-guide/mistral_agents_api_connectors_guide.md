---
title: "Mistral Agents API: Connectors and Built-in Tools Guide"
description: "Web search, code interpreter, image generation, document libraries and MCP connectors in Mistral's Agents and Conversations APIs, with mistralai Python examples."
framework: "mistral-agents-api"
language: "python"
sidebar:
  label: "Built-in tools and connectors"
  order: 2
---

**Verified against `mistralai` %%version=mistralai%%** (checked %%versions-checked%%). Tool shapes on this page were validated against the SDK's request models.

Mistral calls its server-side tools "connectors". They run on Mistral's infrastructure: you declare them on an agent (or a conversation), and the results appear as `tool.execution` entries in the conversation output. Since SDK v3 they are accepted only by the Agents and Conversations APIs, not by `chat.complete()`.

Status: the tools themselves are part of the beta Agents and Conversations APIs; registered MCP connectors (section 5) are a public preview, so their API can change.

| `type` | Purpose | Fields besides `type` |
|---|---|---|
| `web_search` | Web search with citations | `tool_configuration` |
| `web_search_premium` | Web search plus news-agency sources | `tool_configuration` |
| `code_interpreter` | Python in a sandbox | `tool_configuration` |
| `image_generation` | Image generation | `tool_configuration` |
| `document_library` | Retrieval over your Libraries | `library_ids`, `tool_configuration` |
| `connector` | A registered MCP server | `connector_id`, `authorization`, `tool_configuration` |

`tool_configuration` has three optional lists: `include` and `exclude` (which of a connector's tools to expose) and `requires_confirmation` (tools that need human approval; see [approving connector and built-in tool calls](/mistral-agents-api-guide/mistral_agents_api_mcp_guide/#approving-connector-and-built-in-tool-calls)). There are no other per-tool settings such as timeouts or result counts.

The examples use this setup and the `text_of()` helper from the [comprehensive guide](/mistral-agents-api-guide/mistral_agents_api_comprehensive_guide/#5-conversations):

```python
import os
from mistralai.client import Mistral
from mistralai.client.models import ConversationResponse, MessageOutputEntry, TextChunk

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

def text_of(response: ConversationResponse) -> str:
    for entry in reversed(response.outputs):
        if isinstance(entry, MessageOutputEntry):
            if isinstance(entry.content, str):
                return entry.content
            return "".join(c.text for c in entry.content if isinstance(c, TextChunk))
    return ""
```

---

## 1. Code interpreter

Runs model-written Python in an isolated sandbox: calculations, data analysis, charts.

```python
analyst = client.beta.agents.create(
    model="mistral-medium-latest",
    name="data-analyst",
    description="Writes and runs Python to answer quantitative questions.",
    instructions="Use the code interpreter for any calculation. Show the code you ran and explain the result.",
    tools=[{"type": "code_interpreter"}],
)

response = client.beta.conversations.start(
    agent_id=analyst.id,
    inputs="What is the compound value of 10,000 EUR at 3.2% for 15 years? Plot it year by year.",
)

for entry in response.outputs:
    if entry.type == "tool.execution":
        print("ran", entry.name, entry.arguments[:200] if isinstance(entry.arguments, str) else entry.arguments)
print(text_of(response))
```

Files the code produces (charts, CSVs) come back as `ToolFileChunk`s in the message content. Download them with the Files API:

```python
from mistralai.client.models import MessageOutputEntry, ToolFileChunk

for entry in response.outputs:
    if isinstance(entry, MessageOutputEntry) and not isinstance(entry.content, str):
        for chunk in entry.content:
            if isinstance(chunk, ToolFileChunk):
                data = client.files.download(file_id=chunk.file_id).read()
                with open(chunk.file_name or f"{chunk.file_id}.png", "wb") as f:
                    f.write(data)
```

Practicalities: the sandbox has no access to your network or files, so pass data in the prompt or have the agent fetch public data with web search first. Ask for printed results as well as charts, so the answer is checkable.

---

## 2. Image generation

```python
designer = client.beta.agents.create(
    model="mistral-medium-latest",
    name="illustrator",
    description="Creates illustrations from descriptions.",
    instructions="Generate one image per request. Describe what you generated in one sentence.",
    tools=[{"type": "image_generation"}],
)

response = client.beta.conversations.start(
    agent_id=designer.id,
    inputs="A flat, two-colour illustration of a lighthouse at dusk, for a blog header.",
)
```

The image arrives as a `ToolFileChunk`; download it exactly as in the code interpreter example. Image generation is billed per image in addition to tokens.

---

## 3. Web search

`web_search` searches the web; `web_search_premium` adds news-agency sources. Both return citations as `ToolReferenceChunk`s interleaved with the text.

```python
from mistralai.client.models import MessageOutputEntry, TextChunk, ToolReferenceChunk

researcher = client.beta.agents.create(
    model="mistral-medium-latest",
    name="researcher",
    description="Answers questions with current, cited information.",
    instructions="Search before answering anything time-sensitive. Prefer primary sources.",
    tools=[{"type": "web_search"}],
)

response = client.beta.conversations.start(
    agent_id=researcher.id,
    inputs="What did the ECB decide at its most recent rate meeting?",
)

for entry in response.outputs:
    if isinstance(entry, MessageOutputEntry) and not isinstance(entry.content, str):
        for chunk in entry.content:
            if isinstance(chunk, TextChunk):
                print(chunk.text, end="")
            elif isinstance(chunk, ToolReferenceChunk):
                print(f" [{chunk.title}]({chunk.url})", end="")
print()
```

At the May 2025 launch Mistral reported SimpleQA accuracy rising from 23% to 75% for Mistral Large and from 22% to 82% for Mistral Medium with web search enabled. Treat those as an indication of the effect, not a figure for current models.

---

## 4. Document libraries (RAG)

Upload documents to a Library and give agents a `document_library` tool pointing at it. Mistral handles chunking, embedding and retrieval.

```python
from pathlib import Path
from mistralai.client.models import File

library = client.beta.libraries.create(
    name="product-handbook",
    description="Product manuals and support policies",
)

docs = [
    client.beta.libraries.documents.upload(
        library_id=library.id,
        file=File(file_name=path.name, content=path.read_bytes()),
    )
    for path in Path("handbook").glob("*.pdf")
]

# Processing is asynchronous; check each document before relying on it
for doc in docs:
    status = client.beta.libraries.documents.status(library_id=library.id, document_id=doc.id)
    print(doc.name, status.process_status)   # "todo", "in_progress", "done" or "error", among others

support = client.beta.agents.create(
    model="mistral-medium-latest",
    name="handbook-support",
    description="Answers questions from the product handbook.",
    instructions="Answer only from the handbook. If it does not cover the question, say so.",
    tools=[{"type": "document_library", "library_ids": [library.id]}],
)

response = client.beta.conversations.start(agent_id=support.id, inputs="What is the refund window for annual plans?")
print(text_of(response))
```

`libraries.create()` also accepts `chunk_size`. Documents can be listed (`documents.list(library_id=..., search=...)`), updated, reprocessed and deleted; `documents.text_content()` returns the extracted text, which is useful for checking what the retriever sees.

---

## 5. MCP connectors

A connector is a remote MCP server registered with Mistral. Once registered, any agent or conversation can use its tools with `{"type": "connector", "connector_id": "<name or id>"}`, and Mistral calls the server for you.

```python
connector = client.beta.connectors.create(
    request={
        "protocol": "mcp",
        "name": "deepwiki",
        "description": "DeepWiki MCP server for exploring public code repositories",
        "server": "https://mcp.deepwiki.com/mcp",
        "visibility": "private",
    }
)

tools = client.beta.connectors.list_tools(connector_id_or_name="deepwiki")

response = client.beta.conversations.start(
    model="mistral-medium-latest",
    inputs="Using deepwiki, explain how the sqlite/sqlite repository is organised.",
    tools=[{
        "type": "connector",
        "connector_id": "deepwiki",
        "tool_configuration": {"include": ["read_wiki_structure", "ask_question"]},
    }],
)
print(text_of(response))
```

For servers that need credentials, store them once and refer to them by name, or start an OAuth flow:

```python
client.beta.connectors.create_credentials(
    connector_id_or_name="internal-tickets",
    consumer_scope="workspace",          # "user", "workspace" or "organization"
    name="default",
    credentials={"headers": {"Authorization": f"Bearer {os.environ['TICKETS_TOKEN']}"}},
    is_default=True,
)

auth = client.beta.connectors.get_auth_url(
    connector_id_or_name="gmail",
    app_return_url="https://myapp.example.com/oauth/callback",
)
print("send the user to", auth.auth_url)
```

To call a connector's tools directly, without the model (useful in tests and for checking credentials), `connectors.mcp_client()` opens an official `mcp` client session through Mistral's Connectors Gateway, using the Mistral client's credentials. It needs `mistralai[agents]`:

```python
import asyncio
from mcp import MCPError
from mistralai.extra.connectors_gateway import ConnectorsGatewayError

async def check_connector() -> None:
    try:
        async with client.beta.connectors.mcp_client(connector_id_or_name="deepwiki") as session:
            listed = await session.list_tools()
            print([t.name for t in listed.tools])
            result = await session.call_tool("read_wiki_structure", {"repoName": "sqlite/sqlite"})
            print(result.content[0].text[:200])
    except ConnectorsGatewayError as e:     # unknown connector, missing credentials, server unreachable
        print("gateway:", e.proxy_error, e.details)
    except MCPError as e:                    # an error returned by the MCP server itself
        print("server:", e)

asyncio.run(check_connector())
```

`connectors.call_tool()` still exists but is deprecated and will be removed. Pass `credentials_name=` to use a stored credential other than the default.

For MCP servers that should run inside your own process or network (local stdio servers, private services), use the client-side MCP support in `mistralai.extra` instead; see the [MCP guide](/mistral-agents-api-guide/mistral_agents_api_mcp_guide/).

---

## 6. Conversation memory

Persistence is not a tool. Conversations are stored server-side by default, so a conversation can be continued days later:

```python
first = client.beta.conversations.start(agent_id=support.id, inputs="I'm on the annual plan, bought in March.")
saved_id = first.conversation_id          # store this with your user record

# ... later ...
later = client.beta.conversations.append(conversation_id=saved_id, inputs="Can I still get a refund?")
print(text_of(later))
```

Use `store=False` on `start()` for requests that must not be kept. Long-term memory across separate conversations is your application's responsibility: store facts yourself and pass them in `instructions` or the first input.

---

## 7. Combining tools

```python
assistant = client.beta.agents.create(
    model="mistral-medium-latest",
    name="analyst-plus",
    description="Researches, computes and illustrates.",
    instructions=(
        "Use web_search for current facts, the code interpreter for numbers and charts, "
        "and the handbook for company policy. Say which tool each claim came from."
    ),
    tools=[
        {"type": "web_search"},
        {"type": "code_interpreter"},
        {"type": "document_library", "library_ids": [library.id]},
    ],
)

response = client.beta.conversations.start(
    agent_id=assistant.id,
    inputs="Compare our refund policy with the EU's statutory 14-day withdrawal right and chart the difference.",
)
print(text_of(response))
```

---

## Best practices

- **Tell the agent when to use each tool.** Instructions such as "search before answering anything time-sensitive" are more reliable than hoping the model infers it.
- **Give each agent few tools.** Several narrow agents connected with [handoffs](/mistral-agents-api-guide/mistral_agents_api_orchestration_guide/) are easier to test than one agent with everything.
- **Inspect `outputs`.** `tool.execution` entries show what ran and with which arguments; log them.
- **Check document processing.** Uploaded documents are not searchable until processing finishes.
- **Watch usage.** `response.usage.connector_tokens` and `response.usage.connectors` (calls per tool, for example `{"web_search": 1}`) report tool consumption separately from your own prompt and completion tokens; see [cost](/mistral-agents-api-guide/mistral_agents_api_production_guide/#cost).

## Security

- Built-in tools run on Mistral's side, so data in prompts and libraries is processed there; check your data-processing agreement and choose the EU or US endpoint (`Mistral(server="eu")`) as required.
- Content fetched by web search or MCP connectors is untrusted input to the model. Use `requires_confirmation` for connector tools that write or send anything.
- Scope connector credentials narrowly (`consumer_scope="user"` where possible) and use `visibility="private"` for connectors that should not be shared across the workspace.
- Restrict a connector's tools with `include` rather than exposing all of them.

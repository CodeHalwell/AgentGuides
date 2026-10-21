---
title: "Semantic Kernel Recipes (Python)"
description: "Ready-to-use Python recipes for Semantic Kernel — Q&A, structured extraction, plugins, RAG with vector stores, multi-agent teams, resilient agents and MCP tools."
framework: semantic-kernel
language: python
---

# Semantic Kernel Recipes (Python)

> Verified against `semantic-kernel` %%version=semantic-kernel%% (checked %%versions-checked%%). Each recipe uses only APIs checked against an installed copy. Model names are examples. For new agent projects consider [Microsoft Agent Framework](/microsoft-agent-framework-guide/python/), Semantic Kernel's successor.

**See also:** [language-agnostic recipes](/semantic-kernel-guide/semantic_kernel_recipes/) · [comprehensive Python guide](/semantic-kernel-guide/python/semantic_kernel_comprehensive_python/)

---

## Table of contents

1. [Simple Q&A](#1-simple-qa)
2. [Summariser with structured output](#2-summariser-with-structured-output)
3. [Custom database plugin](#3-custom-database-plugin)
4. [Document Q&A with RAG](#4-document-qa-with-rag)
5. [Research and writing team](#5-research-and-writing-team)
6. [Resilient agent with monitoring](#6-resilient-agent-with-monitoring)
7. [MCP tools](#7-mcp-tools)
8. [Vector search with filters](#8-vector-search-with-filters)

---

## 1. Simple Q&A

```python
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelArguments

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(ai_model_id="gpt-5-mini"))   # reads OPENAI_API_KEY

qa = kernel.add_function(
    plugin_name="qa",
    function_name="answer",
    prompt=(
        "Answer the question accurately and concisely. "
        "If you are not sure, say so.\n\nQuestion: {{$question}}\nAnswer:"
    ),
)

async def main():
    for question in ["What is Semantic Kernel?", "How does function calling work?"]:
        result = await kernel.invoke(qa, KernelArguments(question=question))
        print(f"Q: {question}\nA: {result}\n")

asyncio.run(main())
```

---

## 2. Summariser with structured output

```python
from pydantic import BaseModel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

class Summary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str          # positive | neutral | negative

async def summarise(text: str) -> Summary:
    settings = OpenAIChatPromptExecutionSettings(response_format=Summary)
    result = await kernel.invoke_prompt(
        "Summarise this text:\n\n{{$text}}",
        arguments=KernelArguments(text=text, settings=settings),
    )
    return Summary.model_validate_json(str(result))
```

`response_format=<Pydantic model>` requests a strict JSON schema, so there is no need to parse JSON out of free text.

---

## 3. Custom database plugin

```python
import sqlite3
from typing import Annotated
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import kernel_function

class CustomerDB:
    def __init__(self, path: str):
        # Read-only connection: the model can only query, never modify
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)

    @kernel_function(description="List customers in a US state.")
    def customers_in_state(self, state: Annotated[str, "Two-letter state code, e.g. CA"]) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, name, email FROM customers WHERE state = ? LIMIT 50", (state.upper(),)
        ).fetchall()
        return [{"id": r[0], "name": r[1], "email": r[2]} for r in rows]

    @kernel_function(description="Get one customer's order count and total spend.")
    def customer_summary(self, customer_id: int) -> dict:
        count, total = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(total), 0) FROM orders WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        return {"orders": count, "total": total}

agent = ChatCompletionAgent(
    service=OpenAIChatCompletion(ai_model_id="gpt-5-mini"),
    name="DataAssistant",
    instructions="Answer questions about customers using the database tools. Never invent data.",
    plugins=[CustomerDB("shop.db")],
)

async def demo():
    response = await agent.get_response(messages="Which customers are in California?")
    print(response.message.content)
```

Prefer narrow, parameterised functions like these over a generic "run this SQL" tool: the model cannot construct a destructive query, and each tool is easy to test.

---

## 4. Document Q&A with RAG

```python
from dataclasses import dataclass
from typing import Annotated
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding
from semantic_kernel.connectors.in_memory import InMemoryCollection
from semantic_kernel.data.vector import vectorstoremodel, VectorStoreField

@vectorstoremodel(collection_name="docs")
@dataclass
class DocChunk:
    id: Annotated[str, VectorStoreField("key")]
    text: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    source: Annotated[str, VectorStoreField("data", is_indexed=True)]
    embedding: Annotated[list[float] | str | None, VectorStoreField("vector", dimensions=1536)] = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = self.text            # embedded automatically on upsert

docs = InMemoryCollection(
    record_type=DocChunk,
    embedding_generator=OpenAITextEmbedding(ai_model_id="text-embedding-3-small"),
)

async def index(chunks: list[tuple[str, str]]):
    await docs.ensure_collection_exists()
    await docs.upsert([DocChunk(id=str(i), text=t, source=s) for i, (t, s) in enumerate(chunks)])

# Expose search as a tool and let the agent decide when to retrieve
kernel.add_function(
    plugin_name="docs",
    function=docs.create_search_function(
        function_name="search",
        description="Search the product documentation. Returns relevant passages.",
        top=4,
    ),
)

rag_agent = ChatCompletionAgent(
    kernel=kernel,
    name="DocsAssistant",
    instructions="Answer only from passages returned by docs-search. Cite the source of each fact.",
)
```

Swap `InMemoryCollection` for `AzureAISearchCollection`, `QdrantCollection`, `PostgresCollection` and so on without changing `DocChunk`. The old `SemanticTextMemory` / `*MemoryStore` API used by earlier versions of this recipe is legacy.

---

## 5. Research and writing team

```python
from semantic_kernel.agents import SequentialOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime

service = OpenAIChatCompletion(ai_model_id="gpt-5-mini")

researcher = ChatCompletionAgent(name="Researcher", description="Finds facts", service=service,
    instructions="List the key facts about the topic, with sources where possible.")
writer = ChatCompletionAgent(name="Writer", description="Drafts", service=service,
    instructions="Write a 200-word article from the research notes.")
editor = ChatCompletionAgent(name="Editor", description="Polishes", service=service,
    instructions="Edit for clarity and accuracy. Return only the final article.")

async def write_article(topic: str) -> str:
    team = SequentialOrchestration(
        members=[researcher, writer, editor],
        agent_response_callback=lambda m: print(f"— {m.name} finished ({len(str(m.content))} chars)"),
    )
    runtime = InProcessRuntime()
    runtime.start()
    try:
        result = await team.invoke(task=topic, runtime=runtime)
        return str((await result.get(timeout=300)).content)
    finally:
        await runtime.stop_when_idle()
```

For iterative review loops, see the custom group-chat manager in [advanced multi-agent](/semantic-kernel-guide/python/semantic_kernel_advanced_multi_agent_python/).

---

## 6. Resilient agent with monitoring

```python
import logging, time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.exceptions import ServiceResponseException

log = logging.getLogger("agent")

class ResilientAgent:
    def __init__(self, agent: ChatCompletionAgent):
        self.agent = agent
        self.threads: dict[str, ChatHistoryAgentThread] = {}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(ServiceResponseException),
        reraise=True,
    )
    async def _call(self, message: str, thread):
        return await self.agent.get_response(messages=message, thread=thread)

    async def chat(self, session_id: str, message: str) -> str:
        start = time.perf_counter()
        response = await self._call(message, self.threads.get(session_id))
        self.threads[session_id] = response.thread
        usage = response.message.metadata.get("usage")
        log.info("session=%s latency_ms=%.0f usage=%s",
                 session_id, (time.perf_counter() - start) * 1000, usage)
        return str(response.message.content)
```

`ServiceResponseException` is what the OpenAI connectors raise for failed calls. Retrying a whole agent turn re-sends the conversation, so keep the retry count low and make tools idempotent.

---

## 7. MCP tools

```python
from semantic_kernel.connectors.mcp import MCPStdioPlugin

async def mcp_demo():
    async with MCPStdioPlugin(
        name="filesystem",
        description="Read files in the project directory",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "./project"],
    ) as fs:
        agent = ChatCompletionAgent(
            service=OpenAIChatCompletion(ai_model_id="gpt-5-mini"),
            name="CodeReader",
            instructions="Answer questions about the project files.",
            plugins=[fs],
        )
        print(await agent.get_response(messages="Summarise README.md"))
```

`MCPStreamableHttpPlugin(name=..., url=...)` connects to remote servers. Earlier versions of this recipe used `MCPClient` / `MCPToolPlugin` classes, which do not exist.

---

## 8. Vector search with filters

```python
async def search_billing_docs(query: str):
    results = await docs.search(query, top=3, filter=lambda r: r.source == "billing-handbook")
    async for hit in results.results:
        print(f"{hit.score:.3f}  {hit.record.source}: {hit.record.text[:80]}")
```

The lambda is translated into each store's native filter syntax; only simple comparisons and boolean combinations are supported, so keep filters to equality and ranges on indexed fields (`is_indexed=True`).

---

**[Back to Python overview](/semantic-kernel-guide/python/)**

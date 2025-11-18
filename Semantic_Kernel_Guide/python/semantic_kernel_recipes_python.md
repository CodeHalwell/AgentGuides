# Semantic Kernel Recipes (Python)

**Ready-to-Use Python Code Examples for Common Patterns**

Last Updated: November 2025
Semantic Kernel Python: 1.38.0+

---

## Overview

This document provides production-ready Python recipes for common Semantic Kernel patterns. Each recipe includes complete, tested code that can be adapted for your use case.

**See Also:** [../semantic_kernel_recipes.md](../semantic_kernel_recipes.md) for additional language-agnostic recipes.

---

## Table of Contents

1. [Basic Recipes](#basic-recipes)
2. [Plugin Recipes](#plugin-recipes)
3. [Memory & RAG Recipes](#memory--rag-recipes)
4. [Multi-Agent Recipes](#multi-agent-recipes)
5. [Production Patterns](#production-patterns)
6. [2025 Features Recipes](#2025-features-recipes)

---

## Basic Recipes

### 1. Simple Q&A System

{% raw %}
```python
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import os

async def simple_qa_system():
    # Initialize kernel
    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            model_id="gpt-4",
            api_key=os.environ["OPENAI_API_KEY"]
        )
    )

    # Create Q&A function
    qa_prompt = """
    Answer the following question concisely and accurately:

    Question: {{$question}}

    Answer:
    """

    qa_function = kernel.create_function_from_prompt(
        qa_prompt,
        function_name="answer_question"
    )

    # Use it
    questions = [
        "What is Semantic Kernel?",
        "How do plugins work in SK?",
        "What are agents in SK?"
    ]

    for question in questions:
        result = await kernel.invoke(qa_function, question=question)
        print(f"Q: {question}")
        print(f"A: {result}\n")

if __name__ == "__main__":
    asyncio.run(simple_qa_system())
```
{% endraw %}

### 2. Content Summarizer with Structured Output

{% raw %}
```python
from pydantic import BaseModel
from typing import List

class Summary(BaseModel):
    main_points: List[str]
    key_takeaway: str
    word_count: int

async def structured_summarizer():
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(...))

    prompt = """
    Summarize the following text and return JSON:

    Text: {{$text}}

    Return JSON with:
    - main_points: list of 3-5 key points
    - key_takeaway: one sentence summary
    - word_count: approximate word count

    JSON:
    """

    function = kernel.create_function_from_prompt(prompt)

    text = """
    Semantic Kernel is an open-source SDK that lets you easily build agents
    that can call your existing code. As a highly extensible SDK, you can use
    Semantic Kernel with models from OpenAI, Azure OpenAI, Hugging Face, and more!
    """

    result = await kernel.invoke(function, text=text)
    summary = Summary.model_validate_json(str(result))

    print(f"Main Points: {summary.main_points}")
    print(f"Takeaway: {summary.key_takeaway}")
    print(f"Word Count: {summary.word_count}")

asyncio.run(structured_summarizer())
```
{% endraw %}

---

## Plugin Recipes

### 3. Custom Database Plugin

```python
from semantic_kernel.functions import kernel_function
from typing import Annotated, List, Dict
import asyncpg

class DatabasePlugin:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None

    async def initialize(self):
        self.pool = await asyncpg.create_pool(self.connection_string)

    @kernel_function(
        name="query_customers",
        description="Query customers by criteria"
    )
    async def query_customers(
        self,
        filter: Annotated[str, "SQL WHERE clause"] = "1=1",
        limit: Annotated[int, "Max results"] = 10
    ) -> Annotated[str, "JSON array of customers"]:
        async with self.pool.acquire() as conn:
            query = f"SELECT * FROM customers WHERE {filter} LIMIT {limit}"
            rows = await conn.fetch(query)
            return json.dumps([dict(row) for row in rows])

    @kernel_function(
        name="update_customer",
        description="Update customer information"
    )
    async def update_customer(
        self,
        customer_id: Annotated[str, "Customer ID"],
        updates: Annotated[str, "JSON of fields to update"]
    ) -> Annotated[str, "Success message"]:
        update_data = json.loads(updates)

        async with self.pool.acquire() as conn:
            set_clause = ", ".join([f"{k} = ${i+2}" for i, k in enumerate(update_data.keys())])
            query = f"UPDATE customers SET {set_clause} WHERE id = $1"
            values = [customer_id] + list(update_data.values())

            await conn.execute(query, *values)
            return f"Customer {customer_id} updated successfully"

# Usage
async def main():
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(...))

    db_plugin = DatabasePlugin(os.environ["DATABASE_URL"])
    await db_plugin.initialize()

    kernel.add_plugin(db_plugin, "Database")

    # Agent can now use database
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="data_agent",
        instructions="You can query and update customer data using the Database plugin."
    )

    result = await agent.invoke("Find all customers in California and show their names")
    print(result)

asyncio.run(main())
```

---

## Memory & RAG Recipes

### 4. Document Q&A with RAG

{% raw %}
```python
from semantic_kernel.memory import SemanticTextMemory
from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchMemoryStore
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding

async def document_qa_rag():
    # Setup
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(model_id="gpt-4", api_key=os.environ["OPENAI_API_KEY"]))

    # Embedding service
    embedding_service = OpenAITextEmbedding(
        model_id="text-embedding-ada-002",
        api_key=os.environ["OPENAI_API_KEY"]
    )

    # Vector store (v1.34)
    memory_store = AzureAISearchMemoryStore(
        search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        api_key=os.environ["AZURE_SEARCH_API_KEY"],
        index_name="documents"
    )

    memory = SemanticTextMemory(
        storage=memory_store,
        embeddings_generator=embedding_service
    )

    # Index documents
    documents = [
        "Semantic Kernel is an SDK for integrating AI into applications.",
        "Plugins extend SK functionality with custom code.",
        "Agents in SK can reason, plan, and use tools.",
        "Vector stores enable semantic search and retrieval."
    ]

    for i, doc in enumerate(documents):
        await memory.save_information(
            collection="kb",
            id=f"doc_{i}",
            text=doc
        )

    # RAG function
    rag_prompt = """
    Using the following context, answer the question:

    Context:
    {{$context}}

    Question: {{$question}}

    Answer:
    """

    rag_function = kernel.create_function_from_prompt(rag_prompt)

    # Query
    question = "How do I extend Semantic Kernel?"

    # Retrieve relevant context
    results = await memory.search(
        collection="kb",
        query=question,
        limit=3,
        min_relevance_score=0.7
    )

    context = "\n".join([r.text for r in results])

    # Generate answer
    answer = await kernel.invoke(
        rag_function,
        context=context,
        question=question
    )

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"\nSources:")
    for r in results:
        print(f"  [{r.relevance:.2f}] {r.text}")

asyncio.run(document_qa_rag())
```
{% endraw %}

---

## Multi-Agent Recipes

### 5. Research & Writing Team

```python
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat

async def research_writing_team():
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(...))

    # Create agents
    researcher = ChatCompletionAgent(
        kernel=kernel,
        name="researcher",
        instructions="""You are a research specialist. Your job is to:
        1. Gather factual information
        2. Verify sources
        3. Provide comprehensive research
        When you complete research, say 'RESEARCH COMPLETE' and hand off to writer.
        """
    )

    writer = ChatCompletionAgent(
        kernel=kernel,
        name="writer",
        instructions="""You are a content writer. Your job is to:
        1. Take research and create engaging content
        2. Ensure clarity and flow
        3. Format appropriately
        When complete, say 'DRAFT COMPLETE' and hand off to editor.
        """
    )

    editor = ChatCompletionAgent(
        kernel=kernel,
        name="editor",
        instructions="""You are an editor. Your job is to:
        1. Review for quality and accuracy
        2. Fix grammar and style
        3. Ensure consistency
        When complete, say 'DONE' to finish.
        """
    )

    # Create group chat
    group_chat = AgentGroupChat(researcher, writer, editor)

    # Process request
    await group_chat.add_user_message(
        "Create a blog post about Semantic Kernel's agent capabilities with examples"
    )

    # Run conversation
    async for message in group_chat.invoke():
        print(f"[{message.name}]: {message.content}\n")

        if "DONE" in message.content:
            break

asyncio.run(research_writing_team())
```

---

## Production Patterns

### 6. Resilient Agent with Monitoring

```python
from opentelemetry import trace
from tenacity import retry, stop_after_attempt, wait_exponential
import time

tracer = trace.get_tracer(__name__)

class ProductionAgent:
    def __init__(self, kernel, name, instructions):
        self.agent = ChatCompletionAgent(
            kernel=kernel,
            name=name,
            instructions=instructions
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def invoke_with_retry(self, message: str):
        return await self.agent.invoke(message)

    async def invoke_with_monitoring(self, message: str):
        with tracer.start_as_current_span("agent_invoke") as span:
            span.set_attribute("agent.name", self.agent.name)
            span.set_attribute("message.length", len(message))

            start = time.time()

            try:
                result = await self.invoke_with_retry(message)

                duration_ms = (time.time() - start) * 1000
                span.set_attribute("duration_ms", duration_ms)
                span.set_attribute("status", "success")

                return result

            except Exception as e:
                span.set_attribute("status", "error")
                span.record_exception(e)
                raise

# Usage
async def main():
    from monitoring import setup_monitoring

    setup_monitoring(Config.from_env())

    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(...))

    agent = ProductionAgent(
        kernel,
        "assistant",
        "You are a helpful assistant"
    )

    result = await agent.invoke_with_monitoring("Hello!")
    print(result)

asyncio.run(main())
```

---

## 2025 Features Recipes

### 7. MCP Client Integration

```python
from semantic_kernel.connectors.mcp import MCPClient, MCPToolPlugin

async def mcp_client_recipe():
    # Setup kernel
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(...))

    # Connect to MCP server
    mcp_plugin = await MCPToolPlugin.from_mcp_server(
        server_url="http://localhost:3000",
        plugin_name="ExternalTools"
    )

    kernel.add_plugin(mcp_plugin)

    # Agent can now use MCP tools
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="mcp_agent",
        instructions="You have access to external tools via MCP. Use them to answer questions."
    )

    result = await agent.invoke("Use the weather tool to get current weather in Tokyo")
    print(result)

asyncio.run(mcp_client_recipe())
```

### 8. Vector Store v1.34 with Metadata Filtering

```python
async def vector_store_v134_recipe():
    # Setup
    embedding_service = OpenAITextEmbedding(...)
    memory_store = AzureAISearchMemoryStore(...)
    memory = SemanticTextMemory(storage=memory_store, embeddings_generator=embedding_service)

    # Index with metadata
    products = [
        {"id": "1", "text": "Laptop - 16GB RAM", "metadata": {"category": "electronics", "price": 999}},
        {"id": "2", "text": "Desk Chair - Ergonomic", "metadata": {"category": "furniture", "price": 299}},
        {"id": "3", "text": "Monitor - 27 inch 4K", "metadata": {"category": "electronics", "price": 499}},
    ]

    for product in products:
        await memory.save_information(
            collection="products",
            id=product["id"],
            text=product["text"],
            additional_metadata=product["metadata"]
        )

    # Search with metadata filter (NEW in v1.34)
    results = await memory.search(
        collection="products",
        query="computer equipment",
        limit=10,
        min_relevance_score=0.7,
        filter={"category": "electronics"}  # Filter by metadata
    )

    for r in results:
        print(f"[{r.relevance:.2f}] {r.text} - ${r.metadata['price']}")

asyncio.run(vector_store_v134_recipe())
```

---

For more recipes, see:
- [Comprehensive Guide](semantic_kernel_comprehensive_python.md) - Detailed examples
- [Production Guide](semantic_kernel_production_python.md) - Production patterns
- [Advanced Multi-Agent Guide](semantic_kernel_advanced_multi_agent_python.md) - Complex multi-agent patterns
- [General Recipes](../semantic_kernel_recipes.md) - Language-agnostic examples

**[Back to Python README](README.md)** | **[Complete Index](GUIDE_INDEX.md)**

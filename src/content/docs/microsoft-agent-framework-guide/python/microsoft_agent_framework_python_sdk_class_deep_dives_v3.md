---
title: "Azure AI Agents SDK (Python) — Class Deep Dives Vol. 3"
description: "Source-verified deep dives into 10 new classes from azure-ai-agents 1.1.0: AsyncFunctionTool, AzureFunctionTool, AzureAISearchTool, VectorStore, ThreadRun, RunStep, ResponseFormatJsonSchema, TruncationObject, MessageAttachment, and AsyncAgentEventHandler."
framework: microsoft-agent-framework
language: python
---

# Azure AI Agents SDK (Python) — Class Deep Dives Vol. 3

**Package:** `azure-ai-agents`  
**Version covered:** 1.1.0  
**Verified against:** installed package at `/usr/local/lib/python3.11/dist-packages/azure/ai/agents/`

This is the third volume of source-verified class deep dives for the `azure-ai-agents` Python SDK. [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sdk_class_deep_dives/) covered `AgentsClient`, `FunctionTool`, `ToolSet`, `CodeInterpreterTool`, `FileSearchTool`, `BingGroundingTool`, `ConnectedAgentTool`, `AgentEventHandler`, `ThreadMessage`, and `OpenApiTool`. This volume covers ten additional classes across async tools, Azure-native integrations, vector store management, run introspection, structured output, and context control.

---

## Table of Contents

1. [AsyncFunctionTool](#asyncfunctiontool)
2. [AzureFunctionTool](#azurefunctiontool)
3. [AzureAISearchTool](#azureaisearchtool)
4. [VectorStore](#vectorstore)
5. [ThreadRun](#threadrun)
6. [RunStep](#runstep)
7. [ResponseFormatJsonSchema](#responseformatjsonschema)
8. [TruncationObject](#truncationobject)
9. [MessageAttachment](#messageattachment)
10. [AsyncAgentEventHandler](#asyncagenteventhandler)
11. [Patterns Combining Multiple Classes](#patterns-combining-multiple-classes)

---

## 1. `AsyncFunctionTool`

**Source:** `azure/ai/agents/models/_patch.py` — `AsyncFunctionTool(BaseFunctionTool)`

`AsyncFunctionTool` is the async counterpart to `FunctionTool`. The key difference is its `execute` method is a coroutine: it `await`s functions that are themselves coroutines and calls functions synchronously when they are not. This lets you register a **mix of sync and async callables** in the same tool instance.

### Import

```python
from azure.ai.agents.models import AsyncFunctionTool
```

### Class signature

```python
class AsyncFunctionTool(BaseFunctionTool):
    async def execute(self, tool_call: RequiredFunctionToolCall) -> Any: ...
```

`AsyncFunctionTool` inherits all constructor and property behaviour from `BaseFunctionTool` (which is also the base of `FunctionTool`). The only difference is `execute` is `async def`.

### Constructor (inherited from `BaseFunctionTool`)

```python
AsyncFunctionTool(functions: Set[Callable])
```

| Parameter | Type | Description |
|---|---|---|
| `functions` | `Set[Callable]` | A set of callables — any mix of sync and `async def` functions |

### Key inherited methods and properties

| Name | Type | Description |
|---|---|---|
| `definitions` | `List[FunctionToolDefinition]` | JSON schema tool definitions derived from function signatures and docstrings |
| `resources` | `ToolResources` | Always an empty `ToolResources()` — function tools carry no server-side resources |
| `add(func)` | method | Register an additional callable after construction |
| `execute(tool_call)` | `async` method | Dispatch the tool call; awaits coroutines, calls sync functions directly |

### When to use `AsyncFunctionTool` vs `FunctionTool`

| Scenario | Use |
|---|---|
| Tool functions are all `def` (sync) | Either works — `FunctionTool` is simpler |
| Tool functions use `await` internally (e.g. `aiohttp`, `asyncpg`) | **`AsyncFunctionTool`** — must be used with the async `AgentsClient` |
| Mix of sync and async functions in same toolset | **`AsyncFunctionTool`** — auto-detects via `inspect.iscoroutinefunction` |
| Using the sync `AgentsClient` | `FunctionTool` only |

### Example: Pure async functions

```python
import asyncio
import os
import httpx
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import AsyncFunctionTool, AsyncToolSet, MessageRole
from azure.identity.aio import DefaultAzureCredential


# Async tool function — uses httpx to call a real API
async def get_weather(city: str) -> str:
    """Fetch current weather for a city from an external API.

    :param city: The city name to get weather for.
    :type city: str
    :return: A summary of current weather conditions.
    :rtype: str
    """
    async with httpx.AsyncClient(timeout=10) as http:
        resp = await http.get(
            "https://wttr.in",
            params={"city": city, "format": "j1"},
        )
        if resp.status_code != 200:
            return f"Weather unavailable for {city} (HTTP {resp.status_code})"
        data = resp.json()
        current = data["current_condition"][0]
        temp_c = current["temp_C"]
        desc = current["weatherDesc"][0]["value"]
        return f"{city}: {temp_c}°C, {desc}"


async def get_time(timezone: str = "UTC") -> str:
    """Return the current time in a given timezone.

    :param timezone: A timezone name such as 'Europe/London' or 'UTC'.
    :type timezone: str
    :return: The current time as an ISO-8601 string.
    :rtype: str
    """
    import datetime, zoneinfo
    tz = zoneinfo.ZoneInfo(timezone)
    now = datetime.datetime.now(tz)
    return now.isoformat()


async def main() -> None:
    tool = AsyncFunctionTool(functions={get_weather, get_time})
    toolset = AsyncToolSet()
    toolset.add(tool)

    async with AgentsClient(
        endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
        credential=DefaultAzureCredential(),
    ) as client:
        agent = await client.create_agent(
            model="gpt-4o",
            name="weather-time-agent",
            instructions="Answer questions about weather and time. Call the appropriate tool.",
            tools=toolset.definitions,
        )
        thread = await client.threads.create()
        await client.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content="What's the weather in London and what time is it there?",
        )

        run = await client.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent.id,
            toolset=toolset,   # pass toolset so the client can dispatch tool calls
        )
        print(f"Run status: {run.status}")

        messages = client.messages.list(thread_id=thread.id)
        async for msg in messages:
            if msg.role == "assistant":
                for tc in msg.text_messages:
                    print(f"Assistant: {tc.text.value}")

        await client.threads.delete(thread.id)
        await client.delete_agent(agent.id)


asyncio.run(main())
```

### Example: Mixed sync + async in one tool

```python
import asyncio, os, json
import aiofiles
from azure.ai.agents.models import AsyncFunctionTool

# Sync function — safe to call in the async event loop; no blocking I/O
def format_currency(amount: float, currency: str = "GBP") -> str:
    """Format a number as a currency string.

    :param amount: The numeric amount.
    :type amount: float
    :param currency: Three-letter ISO currency code.
    :type currency: str
    :return: Formatted currency string.
    :rtype: str
    """
    symbols = {"GBP": "£", "USD": "$", "EUR": "€"}
    sym = symbols.get(currency.upper(), currency)
    return f"{sym}{amount:,.2f}"


# Async function — reads a config file without blocking
async def read_config(key: str) -> str:
    """Read a value from the agent configuration file.

    :param key: The configuration key to look up.
    :type key: str
    :return: The string value for the key, or an error if not found.
    :rtype: str
    """
    async with aiofiles.open("/etc/agent-config.json") as f:
        config = json.loads(await f.read())
    return str(config.get(key, f"Key '{key}' not found"))


# AsyncFunctionTool handles both — no manual dispatch needed
tool = AsyncFunctionTool(functions={format_currency, read_config})
print(tool.definitions)  # Both functions appear as tool definitions
```

### Error handling behaviour

When a tool function raises an exception, `execute` catches it, logs the error, and returns a JSON string `{"error": "<message>"}` back to the agent. This allows the agent to **self-correct** — it sees the error in the tool output and can try a different approach or report it to the user.

```python
# The return path on error (from source):
# return json.dumps({"error": f"Error executing function '{tool_call.function.name}': {e}"})
```

---

## 2. `AzureFunctionTool`

**Source:** `azure/ai/agents/models/_patch.py` — `AzureFunctionTool(Tool[AzureFunctionToolDefinition])`

`AzureFunctionTool` enables agents to call **Azure Functions via Azure Storage Queues**. Instead of executing function code in the Python process, the agent writes a job to an input queue; your Azure Function reads it, processes it, and writes the result to an output queue. The Agents service polls the output queue automatically.

This is the right pattern when:
- You need serverless compute that scales independently of the agent process.
- Tool logic is too heavy for an in-process function.
- You want to reuse existing Azure Functions as agent capabilities.

### Import

```python
from azure.ai.agents.models import (
    AzureFunctionTool,
    AzureFunctionStorageQueue,
)
```

### Supporting classes

#### `AzureFunctionStorageQueue`

Describes a single Azure Storage Queue used as a binding.

```python
AzureFunctionStorageQueue(
    *,
    storage_service_endpoint: str,   # e.g. "https://<account>.queue.core.windows.net"
    queue_name: str,                  # Name of the queue, e.g. "agent-input"
)
```

| Field | Description |
|---|---|
| `storage_service_endpoint` | Full URI to the Storage Queue service (serialised as `queue_service_endpoint` in JSON) |
| `queue_name` | The queue name within that storage account |

#### `AzureFunctionBinding` (used internally)

Wraps an `AzureFunctionStorageQueue` with `type = "storage_queue"`. `AzureFunctionTool` creates this automatically — you do not normally construct it yourself.

#### `AzureFunctionDefinition` (used internally)

Holds the `FunctionDefinition` (name, description, parameters schema) plus the two bindings. Also constructed automatically.

### `AzureFunctionTool` constructor

```python
AzureFunctionTool(
    name: str,
    description: str,
    parameters: Dict[str, Any],   # JSON Schema object for the function parameters
    input_queue: AzureFunctionStorageQueue,
    output_queue: AzureFunctionStorageQueue,
)
```

| Parameter | Description |
|---|---|
| `name` | Function name — used as the tool name the model will call |
| `description` | Natural-language description for the model |
| `parameters` | JSON Schema `object` describing the input; same format as `FunctionTool` |
| `input_queue` | Queue where the Agents service writes the call payload |
| `output_queue` | Queue where your Azure Function writes its result |

### Properties

| Property | Return type | Description |
|---|---|---|
| `definitions` | `List[AzureFunctionToolDefinition]` | Tool definition list to pass to `create_agent(tools=...)` |
| `resources` | `ToolResources` | Always `ToolResources()` — queue credentials are implicit from the storage endpoint |

### `execute` method

`execute(tool_call)` is a no-op (`pass`). The Agents service handles the queue polling loop server-side — your Python code does not need to do anything extra at call time.

### Example: End-to-end Azure Function integration

This example assumes you have:
- An Azure Storage Account at `https://mystore.queue.core.windows.net`
- Queues named `inventory-input` and `inventory-output`
- An Azure Function triggered by `inventory-input` that processes requests and writes results to `inventory-output`

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    AzureFunctionTool,
    AzureFunctionStorageQueue,
    MessageRole,
)
from azure.identity import DefaultAzureCredential

STORAGE_ENDPOINT = "https://mystore.queue.core.windows.net"

# Describe the queues
input_queue = AzureFunctionStorageQueue(
    storage_service_endpoint=STORAGE_ENDPOINT,
    queue_name="inventory-input",
)
output_queue = AzureFunctionStorageQueue(
    storage_service_endpoint=STORAGE_ENDPOINT,
    queue_name="inventory-output",
)

# Define the tool — the schema tells the model what arguments to pass
inventory_tool = AzureFunctionTool(
    name="check_inventory",
    description=(
        "Check current stock levels for a given product SKU. "
        "Returns available quantity and warehouse location."
    ),
    parameters={
        "type": "object",
        "properties": {
            "sku": {
                "type": "string",
                "description": "The product SKU code, e.g. 'WIDGET-42'.",
            },
            "warehouse": {
                "type": "string",
                "description": "Optional warehouse ID to narrow the check.",
            },
        },
        "required": ["sku"],
    },
    input_queue=input_queue,
    output_queue=output_queue,
)

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

agent = client.create_agent(
    model="gpt-4o",
    name="inventory-agent",
    instructions=(
        "You help warehouse staff check stock levels. "
        "Always use the check_inventory tool to get accurate data."
    ),
    tools=inventory_tool.definitions,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="How many WIDGET-42 units do we have in stock?",
)

# create_and_process polls until the run completes — including waiting for
# the Azure Function to process the queue message and post the result
run = client.runs.create_and_process(
    thread_id=thread.id,
    agent_id=agent.id,
)
print(f"Run status: {run.status}")

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(f"Agent: {tc.text.value}")

client.threads.delete(thread.id)
client.delete_agent(agent.id)
client.close()
```

### Azure Function side (Node.js example)

```javascript
// index.js — Azure Function triggered by storage queue
const { QueueClient } = require("@azure/storage-queue");

module.exports = async function (context, queueItem) {
    // queueItem is the JSON payload from the Agents service
    const { sku, warehouse } = JSON.parse(
        Buffer.from(queueItem, "base64").toString("utf-8")
    );

    // Your business logic here
    const result = await lookupInventory(sku, warehouse);

    // Write result to output queue — Agents service reads this
    const outputClient = new QueueClient(
        process.env.STORAGE_CONNECTION,
        "inventory-output"
    );
    await outputClient.sendMessage(
        Buffer.from(JSON.stringify(result)).toString("base64")
    );
};
```

### Gotchas

- **Queue message format:** The Agents service writes a base64-encoded JSON payload to the input queue. Your Azure Function must decode it as shown above.
- **Result timeout:** If the output queue does not receive a result within the run's `expires_at` window, the run status becomes `expired`.
- **Managed identity:** Both the Agents service and your Azure Function need `Storage Queue Data Contributor` on the storage account. The simplest approach is to grant this to the Azure AI project's managed identity.
- **Parallel calls:** If the model calls the same tool multiple times in one turn, each call gets its own message in the input queue — your Azure Function is invoked once per message.

---

## 3. `AzureAISearchTool`

**Source:** `azure/ai/agents/models/_patch.py` — `AzureAISearchTool(Tool[AzureAISearchToolDefinition])`

`AzureAISearchTool` connects an agent to an existing **Azure AI Search** (formerly Cognitive Search) index. The search is executed **server-side** by the Agents service — the Python SDK only communicates the connection parameters and query configuration; it never calls the index directly.

### Import

```python
from azure.ai.agents.models import AzureAISearchTool, AzureAISearchQueryType
```

### Constructor

```python
AzureAISearchTool(
    index_connection_id: str,
    index_name: str,
    query_type: AzureAISearchQueryType = AzureAISearchQueryType.SIMPLE,
    filter: str = "",
    top_k: int = 5,
    index_asset_id: str = "",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `index_connection_id` | `str` | required | The Azure AI project connection ID that references the Azure AI Search resource |
| `index_name` | `str` | required | Name of the index within the Azure AI Search service |
| `query_type` | `AzureAISearchQueryType` | `SIMPLE` | How the query is executed — see table below |
| `filter` | `str` | `""` | OData filter expression (e.g. `"category eq 'electronics'"`) |
| `top_k` | `int` | `5` | Number of documents to retrieve and pass as context to the model |
| `index_asset_id` | `str` | `""` | Optional asset ID for specific index versions |

### `AzureAISearchQueryType` values

| Value | Description |
|---|---|
| `SIMPLE` | Default. Keyword-based simple query syntax |
| `SEMANTIC` | Semantic ranking — best for natural-language questions |
| `VECTOR` | Pure vector similarity search — requires vector fields |
| `VECTOR_SIMPLE_HYBRID` | Combines keyword + vector search |
| `VECTOR_SEMANTIC_HYBRID` | Combines semantic + vector search (highest quality, higher cost) |

### Properties

| Property | Return type | Description |
|---|---|---|
| `definitions` | `List[AzureAISearchToolDefinition]` | Pass to `create_agent(tools=...)` |
| `resources` | `ToolResources` | `ToolResources(azure_ai_search=AzureAISearchToolResource(index_list=[...]))` — pass to `create_agent(tool_resources=...)` |

> **Both `definitions` and `resources` must be passed** when creating the agent, unlike `FunctionTool` which only uses `definitions`.

### Getting the `index_connection_id`

In the Azure AI Studio portal: **Project → Settings → Connections → [your search resource] → Copy connection ID**. It follows the format:
```
/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/<project>/connections/<connection-name>
```

### Example: Semantic search over a product catalogue

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    AzureAISearchTool,
    AzureAISearchQueryType,
    MessageRole,
)
from azure.identity import DefaultAzureCredential

# Connection parameters — retrieved from Azure AI Studio
CONNECTION_ID = os.environ["AZURE_SEARCH_CONNECTION_ID"]
INDEX_NAME = "product-catalogue"

search_tool = AzureAISearchTool(
    index_connection_id=CONNECTION_ID,
    index_name=INDEX_NAME,
    query_type=AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
    top_k=8,
    filter="in_stock eq true",  # only show in-stock products
)

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

agent = client.create_agent(
    model="gpt-4o",
    name="product-search-agent",
    instructions=(
        "You are a shopping assistant. Search the product catalogue and give "
        "concise, accurate answers about product availability and features."
    ),
    tools=search_tool.definitions,
    tool_resources=search_tool.resources,  # ← required for AzureAISearchTool
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="Do you have any waterproof running shoes under £80?",
)
run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(tc.text.value)

client.threads.delete(thread.id)
client.delete_agent(agent.id)
client.close()
```

### Example: Filtering by date range with OData

```python
from datetime import datetime, timedelta

# Only surface documents updated in the last 90 days
cutoff = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%dT00:00:00Z")

news_search = AzureAISearchTool(
    index_connection_id=CONNECTION_ID,
    index_name="news-articles",
    query_type=AzureAISearchQueryType.SEMANTIC,
    filter=f"published_date ge {cutoff}",
    top_k=5,
)
```

### Example: Multiple search tools (different indices)

```python
# One tool per index — agent chooses which to call based on the question
product_search = AzureAISearchTool(
    index_connection_id=CONNECTION_ID,
    index_name="products",
    query_type=AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
    top_k=5,
)
docs_search = AzureAISearchTool(
    index_connection_id=CONNECTION_ID,
    index_name="documentation",
    query_type=AzureAISearchQueryType.SEMANTIC,
    top_k=3,
)

# Merge definitions and resources
combined_tools = product_search.definitions + docs_search.definitions
# Resources: each tool creates its own index entry in AzureAISearchToolResource
# Pass both resources — the SDK merges index_list automatically when using ToolSet
from azure.ai.agents.models import ToolSet
toolset = ToolSet()
toolset.add(product_search)
toolset.add(docs_search)

agent = client.create_agent(
    model="gpt-4o",
    name="multi-index-agent",
    instructions="Search products OR documentation depending on the user question.",
    tools=toolset.definitions,
    tool_resources=toolset.resources,
)
```

### Gotchas

- `execute()` is a no-op — the Agents service performs the search on its own.
- The `index_connection_id` must be a connection **in the same Azure AI project** as your Agents endpoint.
- `top_k` counts the documents given to the model as context — higher values increase accuracy but also token usage.
- For vector and hybrid modes, your index must have a `vectorSearch` configuration and a searchable vector field.

---

## 4. `VectorStore`

**Source:** `azure/ai/agents/models/_models.py` — `VectorStore(_Model)`

`VectorStore` is the **read model** returned by the vector store API endpoints. You never construct it yourself — the `client.vector_stores.*` methods return it. It represents a fully managed, server-side store of embedded document chunks used by `FileSearchTool`.

### Import

```python
from azure.ai.agents.models import (
    VectorStore,
    VectorStoreStatus,
    VectorStoreExpirationPolicy,
    VectorStoreExpirationPolicyAnchor,
    VectorStoreFileCount,
    VectorStoreStaticChunkingStrategyRequest,
    VectorStoreAutoChunkingStrategyRequest,
    VectorStoreStaticChunkingStrategyOptions,
)
```

### Key fields

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Resource ID — use this in `FileSearchTool(vector_store_ids=[...])` |
| `name` | `str` | Display name |
| `status` | `VectorStoreStatus` | `"expired"`, `"in_progress"`, or `"completed"` |
| `file_counts` | `VectorStoreFileCount` | Counts of files at each processing status |
| `usage_bytes` | `int` | Storage consumed by this vector store |
| `expires_after` | `VectorStoreExpirationPolicy \| None` | Expiry policy if set |
| `expires_at` | `datetime \| None` | Computed expiry timestamp |
| `last_active_at` | `datetime` | Last time the store was queried or modified |
| `metadata` | `Dict[str, str]` | Up to 16 arbitrary key-value tags |
| `created_at` | `datetime` | Creation timestamp |

### `VectorStoreStatus` values

```python
class VectorStoreStatus(str, Enum):
    EXPIRED    = "expired"      # Store has expired; files are gone
    IN_PROGRESS = "in_progress" # Still processing uploaded files
    COMPLETED  = "completed"    # Ready for use with FileSearchTool
```

### `VectorStoreFileCount` fields

```python
file_counts.in_progress  # int — currently being embedded
file_counts.completed    # int — ready to search
file_counts.failed       # int — failed during processing
file_counts.cancelled    # int — cancelled
file_counts.total        # int — sum of all
```

### Operations: `client.vector_stores`

| Method | Description |
|---|---|
| `create(file_ids, name, expires_after, chunking_strategy, metadata)` | Create a vector store; files start processing immediately |
| `create_and_poll(...)` | Create and block until status is `completed` or `expired` |
| `get(vector_store_id)` | Fetch current state |
| `list(limit, order, before)` | Paginated list of all vector stores |
| `modify(vector_store_id, name, expires_after, metadata)` | Update name, expiry, or metadata |
| `delete(vector_store_id)` | Delete the store and free storage |

### `VectorStoreExpirationPolicy`

```python
VectorStoreExpirationPolicy(
    anchor="last_active_at",  # Only supported anchor
    days=7,                   # Expire 7 days after last activity
)
```

The store resets its `expires_at` timestamp every time it's queried. A 7-day policy means the store stays alive as long as it's used at least once a week.

### Chunking strategies

| Strategy class | Type string | Description |
|---|---|---|
| `VectorStoreAutoChunkingStrategyRequest` | `"auto"` | Default — SDK picks optimal chunk size |
| `VectorStoreStaticChunkingStrategyRequest` | `"static"` | You control `max_chunk_size_tokens` and `chunk_overlap_tokens` |

```python
VectorStoreStaticChunkingStrategyRequest(
    static=VectorStoreStaticChunkingStrategyOptions(
        max_chunk_size_tokens=512,
        chunk_overlap_tokens=64,
    )
)
```

### Example: Full vector store lifecycle

```python
import os, time
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    FilePurpose,
    FileSearchTool,
    VectorStoreExpirationPolicy,
    VectorStoreExpirationPolicyAnchor,
    VectorStoreStaticChunkingStrategyRequest,
    VectorStoreStaticChunkingStrategyOptions,
    MessageRole,
)
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# 1. Upload files to the Agents file store
with open("product-manual.pdf", "rb") as f:
    file_obj = client.files.upload(file=f, purpose=FilePurpose.AGENTS)
print(f"Uploaded file: {file_obj.id}")

# 2. Create a vector store and wait for processing to complete
store = client.vector_stores.create_and_poll(
    file_ids=[file_obj.id],
    name="product-manuals",
    expires_after=VectorStoreExpirationPolicy(
        anchor=VectorStoreExpirationPolicyAnchor.LAST_ACTIVE_AT,
        days=30,
    ),
    chunking_strategy=VectorStoreStaticChunkingStrategyRequest(
        static=VectorStoreStaticChunkingStrategyOptions(
            max_chunk_size_tokens=800,
            chunk_overlap_tokens=100,
        )
    ),
)
print(f"Vector store: {store.id}, status: {store.status}")
print(f"Files — completed: {store.file_counts.completed}, failed: {store.file_counts.failed}")

# 3. Wire the vector store into a FileSearchTool
search_tool = FileSearchTool(vector_store_ids=[store.id])

# 4. Create the agent
agent = client.create_agent(
    model="gpt-4o",
    name="manual-assistant",
    instructions="Answer questions using the product manual. Cite page numbers where possible.",
    tools=search_tool.definitions,
    tool_resources=search_tool.resources,
)

# 5. Run a query
thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="What are the safety warnings for the power adapter?",
)
run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(tc.text.value)

# 6. Cleanup
client.threads.delete(thread.id)
client.delete_agent(agent.id)
client.vector_stores.delete(store.id)
client.files.delete(file_obj.id)
client.close()
```

### Example: Inspect and refresh vector store state

```python
store = client.vector_stores.get(store_id)
print(f"Status: {store.status}")
print(f"Size: {store.usage_bytes / 1024:.1f} KB")
print(f"Files: {store.file_counts.completed} ready, {store.file_counts.in_progress} pending")
print(f"Expires: {store.expires_at or 'never'}")

# Renew expiry without re-creating
if store.status == "completed":
    from azure.ai.agents.models import VectorStoreExpirationPolicy, VectorStoreExpirationPolicyAnchor
    client.vector_stores.modify(
        store.id,
        expires_after=VectorStoreExpirationPolicy(
            anchor=VectorStoreExpirationPolicyAnchor.LAST_ACTIVE_AT,
            days=60,
        ),
    )
```

---

## 5. `ThreadRun`

**Source:** `azure/ai/agents/models/_models.py` — `ThreadRun(_Model)`

`ThreadRun` represents a **single invocation** of an agent on a thread. It is the object returned by `client.runs.create(...)`, `client.runs.get(...)`, `client.runs.create_and_process(...)`, and passed into `AgentEventHandler.on_thread_run()` during streaming.

### Import

```python
from azure.ai.agents.models import ThreadRun, RunStatus
```

### Key fields

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Run identifier |
| `thread_id` | `str` | The thread this run belongs to |
| `agent_id` | `str` | Agent that ran (serialised as `assistant_id` in the JSON API) |
| `status` | `RunStatus` | Current status — see below |
| `required_action` | `RequiredAction \| None` | Set when `status == "requires_action"`; contains tool call info |
| `last_error` | `RunError \| None` | Details when `status == "failed"` |
| `model` | `str` | Model deployment used |
| `instructions` | `str` | System prompt used for this run (may be overridden per-run) |
| `tools` | `List[ToolDefinition]` | Tools available to this run |
| `usage` | `RunCompletionUsage \| None` | Token usage; `None` while in non-terminal state |
| `truncation_strategy` | `TruncationObject` | Context truncation settings |
| `parallel_tool_calls` | `bool` | Whether tools were allowed to run concurrently |
| `temperature` | `float \| None` | Sampling temperature override |
| `top_p` | `float \| None` | Nucleus sampling override |
| `max_prompt_tokens` | `int \| None` | Prompt token budget |
| `max_completion_tokens` | `int \| None` | Completion token budget |
| `created_at` | `datetime` | When the run was queued |
| `started_at` | `datetime \| None` | When execution began |
| `completed_at` | `datetime \| None` | When it finished |
| `failed_at` | `datetime \| None` | When it failed |
| `expires_at` | `datetime` | When it will be abandoned if still running |
| `metadata` | `Dict[str, str]` | Up to 16 key-value tags |

### `RunStatus` enum

```python
class RunStatus(str, Enum):
    QUEUED          = "queued"           # Waiting for compute
    IN_PROGRESS     = "in_progress"      # Currently executing
    REQUIRES_ACTION = "requires_action"  # Paused — needs tool outputs
    CANCELLING      = "cancelling"       # Cancel requested, not yet done
    CANCELLED       = "cancelled"        # Successfully cancelled
    FAILED          = "failed"           # Terminal — check last_error
    COMPLETED       = "completed"        # Terminal — check messages for response
    EXPIRED         = "expired"          # Timed out
```

### `RunCompletionUsage` fields

```python
run.usage.prompt_tokens      # Tokens used in the prompt
run.usage.completion_tokens  # Tokens in the model's completion
run.usage.total_tokens       # prompt + completion
```

### `client.runs` operations

| Method | Description |
|---|---|
| `create(thread_id, agent_id, ...)` | Start a run; returns immediately with status `queued` |
| `create_and_process(thread_id, agent_id, ...)` | Start and poll until terminal state; handles tool calls if `toolset` provided |
| `get(thread_id, run_id)` | Fetch the current state of a run |
| `list(thread_id, ...)` | List all runs on a thread |
| `update(thread_id, run_id, metadata=...)` | Update run metadata |
| `cancel(thread_id, run_id)` | Request cancellation |
| `submit_tool_outputs(thread_id, run_id, tool_outputs=[...])` | Submit tool results when `requires_action` |
| `stream(thread_id, agent_id, ...)` | Start a streaming run |

### Example: Manual polling with token budget

```python
import os, time
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    MessageRole, RunStatus, TruncationObject, TruncationStrategy,
)
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

agent = client.create_agent(
    model="gpt-4o",
    name="budget-aware-agent",
    instructions="You are a concise assistant. Use no more than 200 words per answer.",
)
thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="Explain the water cycle in detail.",
)

# Create run with explicit token and truncation settings
run = client.runs.create(
    thread_id=thread.id,
    agent_id=agent.id,
    max_prompt_tokens=4096,
    max_completion_tokens=512,
    truncation_strategy=TruncationObject(
        type=TruncationStrategy.LAST_MESSAGES,
        last_messages=10,   # Only use the last 10 messages as context
    ),
    temperature=0.3,
)
print(f"Run created: {run.id}, status: {run.status}")

# Manual polling loop
while run.status in (RunStatus.QUEUED, RunStatus.IN_PROGRESS):
    time.sleep(1)
    run = client.runs.get(thread_id=thread.id, run_id=run.id)
    print(f"  ... {run.status}")

# Inspect terminal state
if run.status == RunStatus.COMPLETED:
    if run.usage:
        print(f"Tokens — prompt: {run.usage.prompt_tokens}, "
              f"completion: {run.usage.completion_tokens}, "
              f"total: {run.usage.total_tokens}")
    for msg in client.messages.list(thread_id=thread.id):
        if msg.role == "assistant":
            for tc in msg.text_messages:
                print(f"\nAgent: {tc.text.value}")
elif run.status == RunStatus.FAILED:
    print(f"Run failed: {run.last_error}")
elif run.status == RunStatus.EXPIRED:
    print("Run expired — consider increasing the expiry window or reducing latency")

client.threads.delete(thread.id)
client.delete_agent(agent.id)
client.close()
```

### Example: Cancelling a run

```python
import threading, time
from azure.ai.agents.models import RunStatus

# Start a long-running task
run = client.runs.create(thread_id=thread.id, agent_id=agent.id)

# Cancel after 5 seconds from a background thread
def cancel_later():
    time.sleep(5)
    client.runs.cancel(thread_id=thread.id, run_id=run.id)

threading.Thread(target=cancel_later, daemon=True).start()

while run.status not in (RunStatus.CANCELLED, RunStatus.COMPLETED, RunStatus.FAILED):
    time.sleep(0.5)
    run = client.runs.get(thread_id=thread.id, run_id=run.id)

print(f"Final status: {run.status}")  # "cancelled" or "completed" (if it finished first)
```

### Example: Per-run instruction and tool override

```python
# Override instructions and tools on a single run without changing the agent
run = client.runs.create_and_process(
    thread_id=thread.id,
    agent_id=agent.id,
    instructions="For this run only, respond only in French.",
    # Restrict tools for a particular run
    tools=[function_tool.definitions[0]],
    temperature=0.1,
    metadata={"run_type": "french_override", "user_id": "u-123"},
)
```

---

## 6. `RunStep`

**Source:** `azure/ai/agents/models/_models.py` — `RunStep(_Model)`

`RunStep` represents a single **atomic unit of work** within a run — either a message-creation step (the agent wrote a message) or a tool-calls step (the agent invoked one or more tools). Inspecting run steps lets you understand exactly what the agent did, including which tool calls it made and what they returned.

### Import

```python
from azure.ai.agents.models import (
    RunStep,
    RunStepType,
    RunStepStatus,
    RunStepDetails,
    RunStepMessageCreationDetails,
    RunStepMessageCreationReference,
    RunStepToolCallDetails,
    RunStepFunctionToolCall,
    RunStepCodeInterpreterToolCall,
    RunStepFileSearchToolCall,
    RunStepAzureAISearchToolCall,
    RunStepCompletionUsage,
    RunAdditionalFieldList,
)
```

### Key fields

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Step identifier |
| `run_id` | `str` | Parent run |
| `thread_id` | `str` | Parent thread |
| `agent_id` | `str` | Agent that produced this step |
| `type` | `RunStepType` | `"message_creation"` or `"tool_calls"` |
| `status` | `RunStepStatus` | `"in_progress"`, `"completed"`, `"failed"`, `"cancelled"`, `"expired"` |
| `step_details` | `RunStepDetails` | The polymorphic payload — see below |
| `last_error` | `RunStepError \| None` | Error info when failed |
| `usage` | `RunStepCompletionUsage \| None` | Token costs for this step; `None` while in-progress |
| `created_at` | `datetime` | When this step was initiated |
| `completed_at` | `datetime \| None` | Completion timestamp |

### `RunStepType` values

```python
RunStepType.MESSAGE_CREATION  # Agent wrote a new ThreadMessage
RunStepType.TOOL_CALLS        # Agent invoked one or more tools
```

### Inspecting `step_details`

The `step_details` field is a discriminated union. Check `step.type` to safely cast:

```python
from azure.ai.agents.models import (
    RunStepType,
    RunStepMessageCreationDetails,
    RunStepToolCallDetails,
    RunStepFunctionToolCall,
    RunStepCodeInterpreterToolCall,
    RunStepFileSearchToolCall,
)

for step in client.run_steps.list(thread_id=thread_id, run_id=run_id):
    if step.type == RunStepType.MESSAGE_CREATION:
        details: RunStepMessageCreationDetails = step.step_details
        print(f"  Message created: {details.message_creation.message_id}")

    elif step.type == RunStepType.TOOL_CALLS:
        details: RunStepToolCallDetails = step.step_details
        for tool_call in details.tool_calls:
            if isinstance(tool_call, RunStepFunctionToolCall):
                print(f"  Function: {tool_call.function.name}")
                print(f"    Args: {tool_call.function.arguments}")
                print(f"    Output: {tool_call.function.output}")

            elif isinstance(tool_call, RunStepCodeInterpreterToolCall):
                print(f"  Code interpreter input:\n{tool_call.code_interpreter.input}")
                for output in tool_call.code_interpreter.outputs:
                    print(f"  Output: {output}")

            elif isinstance(tool_call, RunStepFileSearchToolCall):
                print(f"  File search executed (result in message)")
```

### `RunAdditionalFieldList` — retrieving file search result content

By default, `file_search` results in run steps do not include the actual retrieved text. Pass the `include` parameter to get it:

```python
steps = client.run_steps.list(
    thread_id=thread.id,
    run_id=run.id,
    include=[RunAdditionalFieldList.FILE_SEARCH_CONTENTS],
)
for step in steps:
    if step.type == RunStepType.TOOL_CALLS:
        for tc in step.step_details.tool_calls:
            if isinstance(tc, RunStepFileSearchToolCall):
                for result in tc.file_search.results:
                    print(f"  Found in: {result.file_id}, score: {result.score}")
                    for block in result.content:
                        print(f"  Text: {block.text}")
```

### `client.run_steps` operations

| Method | Signature | Description |
|---|---|---|
| `get` | `(thread_id, run_id, step_id, include=[...])` | Get a single step |
| `list` | `(thread_id, run_id, include=[...], limit, order, before)` | Paginated list of steps |

### Example: Audit trail — log all tool calls from a run

```python
def audit_run(client, thread_id: str, run_id: str) -> None:
    """Print a structured audit log of every action taken in a run."""
    steps = list(client.run_steps.list(thread_id=thread_id, run_id=run_id))
    steps.sort(key=lambda s: s.created_at)

    total_prompt = total_completion = 0

    for i, step in enumerate(steps, 1):
        print(f"\nStep {i}: {step.type} | {step.status} | {step.created_at:%H:%M:%S}")

        if step.usage:
            total_prompt += step.usage.prompt_tokens
            total_completion += step.usage.completion_tokens

        if step.type == RunStepType.MESSAGE_CREATION:
            msg_id = step.step_details.message_creation.message_id
            print(f"  → Created message {msg_id}")

        elif step.type == RunStepType.TOOL_CALLS:
            for tc in step.step_details.tool_calls:
                if isinstance(tc, RunStepFunctionToolCall):
                    print(f"  → Called function '{tc.function.name}'")
                    print(f"     Args:   {tc.function.arguments}")
                    print(f"     Result: {tc.function.output}")
                elif isinstance(tc, RunStepCodeInterpreterToolCall):
                    lines = tc.code_interpreter.input.splitlines()
                    print(f"  → Code interpreter ({len(lines)} lines)")
                    print(f"     Outputs: {len(tc.code_interpreter.outputs)}")

    print(f"\nTotal tokens — prompt: {total_prompt}, completion: {total_completion}")
```

---

## 7. `ResponseFormatJsonSchema`

**Source:** `azure/ai/agents/models/_models.py` — `ResponseFormatJsonSchema(_Model)`

`ResponseFormatJsonSchema` defines a **strict JSON output schema** for the model. When a run uses this response format, the model is constrained to return a JSON object that matches the provided schema exactly. This is the Azure AI Agents equivalent of OpenAI's "Structured Outputs" feature.

### Import

```python
from azure.ai.agents.models import (
    ResponseFormatJsonSchema,
    ResponseFormatJsonSchemaType,
)
```

### `ResponseFormatJsonSchema` constructor

```python
ResponseFormatJsonSchema(
    *,
    name: str,                # Identifier for the schema
    schema: Any,              # JSON Schema dict
    description: str | None = None,  # Helps the model understand the format
)
```

| Parameter | Description |
|---|---|
| `name` | Short identifier used in the API — must match `[a-zA-Z0-9_-]{1,64}` |
| `schema` | A JSON Schema `dict` describing the expected output object |
| `description` | Optional natural-language hint for the model |

### `ResponseFormatJsonSchemaType` — the wrapper

To actually pass a JSON schema response format to a run or agent, you wrap `ResponseFormatJsonSchema` in `ResponseFormatJsonSchemaType`:

```python
ResponseFormatJsonSchemaType(
    json_schema=ResponseFormatJsonSchema(
        name="my_schema",
        schema={...},
        description="...",
    )
)
```

This produces `{"type": "json_schema", "json_schema": {...}}` in the API payload.

### Using `response_format` on a run

Pass `ResponseFormatJsonSchemaType` to `create_agent`, `create_thread_and_run`, or `runs.create_and_process` via the `response_format` parameter.

### Example: Extracting structured data from unstructured text

```python
import json, os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    MessageRole,
    ResponseFormatJsonSchema,
    ResponseFormatJsonSchemaType,
)
from azure.identity import DefaultAzureCredential

# Define what we want back
RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {
            "type": "string",
            "description": "Name of the merchant or vendor",
        },
        "date": {
            "type": "string",
            "description": "Purchase date in YYYY-MM-DD format",
        },
        "total_amount": {
            "type": "number",
            "description": "Total amount charged in the receipt currency",
        },
        "currency": {
            "type": "string",
            "description": "Three-letter ISO 4217 currency code",
        },
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["description", "amount"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["vendor", "date", "total_amount", "currency", "line_items"],
    "additionalProperties": False,
}

response_format = ResponseFormatJsonSchemaType(
    json_schema=ResponseFormatJsonSchema(
        name="receipt_extraction",
        schema=RECEIPT_SCHEMA,
        description=(
            "Extract receipt information into a structured JSON object. "
            "Parse all line items individually."
        ),
    )
)

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

agent = client.create_agent(
    model="gpt-4o",
    name="receipt-extractor",
    instructions=(
        "You extract structured data from receipt text. "
        "Return ONLY valid JSON matching the provided schema."
    ),
    response_format=response_format,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content=(
        "Extract data from this receipt:\n\n"
        "Tesco Express — 14 Jan 2026\n"
        "Milk 2L          £1.25\n"
        "Bread wholemeal  £1.60\n"
        "Eggs (6)         £2.10\n"
        "TOTAL            £4.95\n"
        "VISA ending 1234"
    ),
)
run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for tc in msg.text_messages:
            data = json.loads(tc.text.value)
            print(json.dumps(data, indent=2))

client.threads.delete(thread.id)
client.delete_agent(agent.id)
client.close()
```

Output (example):
```json
{
  "vendor": "Tesco Express",
  "date": "2026-01-14",
  "total_amount": 4.95,
  "currency": "GBP",
  "line_items": [
    {"description": "Milk 2L", "amount": 1.25},
    {"description": "Bread wholemeal", "amount": 1.60},
    {"description": "Eggs (6)", "amount": 2.10}
  ]
}
```

### Example: Structured output with function tools

When using `response_format=ResponseFormatJsonSchemaType(...)`, the agent can still call function tools on its way to generating the structured response. The schema only constrains the **final assistant message**, not tool call arguments.

```python
def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Fetch the current exchange rate between two currencies.

    :param from_currency: Source currency code (e.g. 'USD').
    :type from_currency: str
    :param to_currency: Target currency code (e.g. 'GBP').
    :type to_currency: str
    :return: Exchange rate as a float.
    :rtype: float
    """
    # Stub — replace with real FX API call
    rates = {"USD_GBP": 0.79, "EUR_GBP": 0.85}
    return rates.get(f"{from_currency}_{to_currency}", 1.0)


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "base_amount": {"type": "number"},
        "base_currency": {"type": "string"},
        "converted_amount": {"type": "number"},
        "target_currency": {"type": "string"},
        "rate_used": {"type": "number"},
    },
    "required": ["base_amount", "base_currency", "converted_amount", "target_currency", "rate_used"],
    "additionalProperties": False,
}

from azure.ai.agents.models import FunctionTool, ToolSet

tool = FunctionTool(functions={get_exchange_rate})
toolset = ToolSet()
toolset.add(tool)

agent = client.create_agent(
    model="gpt-4o",
    name="fx-agent",
    instructions="Convert currencies and return the result as structured JSON.",
    tools=toolset.definitions,
    response_format=ResponseFormatJsonSchemaType(
        json_schema=ResponseFormatJsonSchema(
            name="fx_conversion",
            schema=SUMMARY_SCHEMA,
        )
    ),
)
```

### Gotchas

- `additionalProperties: false` is strongly recommended — without it the model may add extra fields that break downstream parsing.
- `required` should list all fields you need — the model may omit optional fields.
- JSON schema format does **not** work with `AgentsResponseFormat(type="json_object")` — that is a separate, less strict mode.
- Not all model deployments support structured output — check Azure AI Studio for model capability.

---

## 8. `TruncationObject`

**Source:** `azure/ai/agents/models/_models.py` — `TruncationObject(_Model)`

`TruncationObject` controls **how the thread history is trimmed** when the accumulated messages exceed the model's context window. You set this per-run via the `truncation_strategy` parameter of `runs.create(...)` or `runs.create_and_process(...)`.

### Import

```python
from azure.ai.agents.models import TruncationObject, TruncationStrategy
```

### Constructor

```python
TruncationObject(
    *,
    type: TruncationStrategy | str,  # "auto" or "last_messages"
    last_messages: int | None = None, # Required when type == "last_messages"
)
```

### `TruncationStrategy` values

| Value | Behaviour |
|---|---|
| `"auto"` | **Default.** The service drops messages from the middle of the thread to fit the model's context window while preserving the most recent messages and the system prompt |
| `"last_messages"` | Only the most recent `last_messages` messages are included in the context, regardless of token count |

### When to use each strategy

| Scenario | Recommended strategy |
|---|---|
| General-purpose agents; you don't know in advance how long threads will get | `auto` |
| Cost control — you want to cap how much history the model sees | `last_messages` with a fixed count |
| Tasks where older context is irrelevant (e.g. step-by-step instructions) | `last_messages: 2` to 5 |
| Tasks where full history is critical (e.g. long document analysis) | `auto` + generous `max_prompt_tokens` |

### Example: Capping context to the last 5 messages

```python
from azure.ai.agents.models import TruncationObject, TruncationStrategy, MessageRole
import os
from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

agent = client.create_agent(
    model="gpt-4o",
    name="customer-support-agent",
    instructions=(
        "You are a customer support agent. Focus on the current issue "
        "without dwelling on old resolved issues."
    ),
)

thread = client.threads.create()

# Simulate a long conversation with many prior messages
prior_messages = [
    ("What's your return policy?", "Our policy allows returns within 30 days."),
    ("Can I return a used item?", "Lightly used items may be returned with a 10% restocking fee."),
    ("How long for a refund?", "Refunds process in 5-7 business days."),
    ("What about international orders?", "International orders cannot be returned."),
]
for user_msg, asst_msg in prior_messages:
    client.messages.create(thread_id=thread.id, role=MessageRole.USER, content=user_msg)
    client.messages.create(thread_id=thread.id, role=MessageRole.ASSISTANT, content=asst_msg)

# New question — run with only the last 5 messages in context
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="I bought a jacket 3 weeks ago. Can I still return it?",
)

run = client.runs.create_and_process(
    thread_id=thread.id,
    agent_id=agent.id,
    truncation_strategy=TruncationObject(
        type=TruncationStrategy.LAST_MESSAGES,
        last_messages=5,   # only the most recent 5 messages (4 injected + 1 new question)
    ),
    max_prompt_tokens=2048,
)
print(f"Tokens used: {run.usage.total_tokens if run.usage else 'N/A'}")

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(f"Agent: {tc.text.value}")

client.threads.delete(thread.id)
client.delete_agent(agent.id)
client.close()
```

### Example: Auto truncation with prompt token budget

```python
run = client.runs.create_and_process(
    thread_id=thread.id,
    agent_id=agent.id,
    truncation_strategy=TruncationObject(type=TruncationStrategy.AUTO),
    max_prompt_tokens=8192,      # hard cap on prompt tokens
    max_completion_tokens=1024,  # hard cap on completion tokens
)
```

### Relationship with `ThreadRun.truncation_strategy`

After a run completes, `run.truncation_strategy` reflects the strategy actually used — useful for confirming that your setting was applied:

```python
run = client.runs.get(thread_id=thread.id, run_id=run.id)
ts = run.truncation_strategy
print(f"Strategy: {ts.type}, last_messages: {ts.last_messages}")
```

---

## 9. `MessageAttachment`

**Source:** `azure/ai/agents/models/_patch.py` — `MessageAttachment(MessageAttachmentGenerated)`

`MessageAttachment` lets you attach **files to individual messages**, rather than to the agent globally. The attached file is available only within the thread where it was added. It can be used with either `FileSearchTool` (for document Q&A) or `CodeInterpreterTool` (for data analysis).

### Import

```python
from azure.ai.agents.models import (
    MessageAttachment,
    FileSearchToolDefinition,
    CodeInterpreterToolDefinition,
    VectorStoreDataSource,
    VectorStoreDataSourceAssetType,
)
```

### Constructor overloads

```python
# For FileSearch (document Q&A)
MessageAttachment(
    *,
    tools: List[FileSearchToolDefinition],
    file_id: str | None = None,       # ID of a previously uploaded file
    data_source: VectorStoreDataSource | None = None,  # or an Azure data source
)

# For CodeInterpreter (data processing)
MessageAttachment(
    *,
    tools: List[CodeInterpreterToolDefinition],
    file_id: str | None = None,
    data_source: VectorStoreDataSource | None = None,
)
```

You must provide **exactly one** of `file_id` or `data_source`, and at least one `tools` entry.

### `VectorStoreDataSource` — attaching Azure data assets

```python
VectorStoreDataSource(
    asset_identifier: str,             # e.g. an Azure Blob Storage URI
    asset_type: VectorStoreDataSourceAssetType,  # URI_ASSET or ID_ASSET
)
```

| `VectorStoreDataSourceAssetType` | Description |
|---|---|
| `URI_ASSET` | An Azure storage URL (e.g. `https://mystore.blob.core.windows.net/docs/report.pdf`) |
| `ID_ASSET` | An Azure ML asset ID |

### Example: Attaching a PDF to a message for file search

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    FilePurpose,
    FileSearchTool,
    FileSearchToolDefinition,
    MessageAttachment,
    MessageRole,
)
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# Upload a file first
with open("Q4-report.pdf", "rb") as f:
    file_obj = client.files.upload(file=f, purpose=FilePurpose.AGENTS)

# Create agent with FileSearch enabled (no pre-built vector store needed)
search_tool = FileSearchTool()
agent = client.create_agent(
    model="gpt-4o",
    name="report-analyst",
    instructions="Analyse the attached report and answer questions about it.",
    tools=search_tool.definitions,
)

thread = client.threads.create()

# Attach the file to this specific message
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="Summarise the key financial highlights from this report.",
    attachments=[
        MessageAttachment(
            file_id=file_obj.id,
            tools=[FileSearchToolDefinition()],
        )
    ],
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(tc.text.value)

# Cleanup
client.threads.delete(thread.id)
client.delete_agent(agent.id)
client.files.delete(file_obj.id)
client.close()
```

### Example: Attaching a CSV for code interpreter analysis

```python
from azure.ai.agents.models import (
    CodeInterpreterTool,
    CodeInterpreterToolDefinition,
    MessageAttachment,
    FilePurpose,
    MessageRole,
)

# Upload CSV
with open("sales-data.csv", "rb") as f:
    csv_file = client.files.upload(file=f, purpose=FilePurpose.AGENTS)

ci_tool = CodeInterpreterTool()
agent = client.create_agent(
    model="gpt-4o",
    name="data-analyst",
    instructions="Analyse data files and produce charts and statistical summaries.",
    tools=ci_tool.definitions,
)

thread = client.threads.create()

# Attach the CSV with CodeInterpreter so the agent can read it as a DataFrame
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="Plot monthly sales trends and calculate the percentage growth month-over-month.",
    attachments=[
        MessageAttachment(
            file_id=csv_file.id,
            tools=[CodeInterpreterToolDefinition()],
        )
    ],
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

# Check for image outputs in messages
for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for content in msg.content:
            if hasattr(content, "image_file"):
                # Download the generated chart
                image_data = client.files.get_content(content.image_file.file_id)
                with open("sales_chart.png", "wb") as out:
                    out.write(image_data)
                print("Chart saved to sales_chart.png")
            elif hasattr(content, "text"):
                print(content.text.value)

client.threads.delete(thread.id)
client.delete_agent(agent.id)
client.files.delete(csv_file.id)
```

### Example: Attaching from Azure Blob Storage (no prior upload)

```python
from azure.ai.agents.models import (
    VectorStoreDataSource,
    VectorStoreDataSourceAssetType,
    MessageAttachment,
    FileSearchToolDefinition,
)

# Reference a file in Azure Blob Storage directly — no upload step needed
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="Review the compliance document and identify any gaps.",
    attachments=[
        MessageAttachment(
            data_source=VectorStoreDataSource(
                asset_identifier=(
                    "https://mycompany.blob.core.windows.net/compliance/policy-2026.pdf"
                ),
                asset_type=VectorStoreDataSourceAssetType.URI_ASSET,
            ),
            tools=[FileSearchToolDefinition()],
        )
    ],
)
```

### Key differences: message attachments vs agent-level tool resources

| | Message attachment | Agent tool resource |
|---|---|---|
| **Scope** | Attached to one message; available for the lifetime of the thread | Attached to all runs; always available |
| **Vector store** | Created automatically; ephemeral | You manage the lifecycle |
| **Best for** | Ad-hoc per-query documents | Permanent knowledge bases |
| **Cleanup** | Automatic | Manual (`vector_stores.delete`) |

---

## 10. `AsyncAgentEventHandler`

**Source:** `azure/ai/agents/models/_patch.py` — `AsyncAgentEventHandler(BaseAsyncAgentEventHandler)`

`AsyncAgentEventHandler` is the async streaming handler for the Azure AI Agents SDK. Subclass it and override the `on_*` methods to react to events as they arrive from the streaming run. It handles tool call dispatching and retry logic automatically.

### Import

```python
from azure.ai.agents.models import AsyncAgentEventHandler
from azure.ai.agents.aio import AgentsClient  # async client required
```

### Class signature

```python
class AsyncAgentEventHandler(BaseAsyncAgentEventHandler):
    def __init__(self) -> None: ...

    def set_max_retry(self, max_retry: int) -> None: ...

    # Override these in your subclass:
    async def on_message_delta(self, delta: MessageDeltaChunk) -> None: ...
    async def on_thread_message(self, message: ThreadMessage) -> None: ...
    async def on_thread_run(self, run: ThreadRun) -> None: ...
    async def on_run_step(self, step: RunStep) -> None: ...
    async def on_run_step_delta(self, delta: RunStepDeltaChunk) -> None: ...
    async def on_error(self, data: str) -> None: ...
    async def on_done(self) -> None: ...
    async def on_unhandled_event(self, event_type: str, event_data: Any) -> None: ...
```

### Event dispatch order (typical run)

```
on_thread_run(status="in_progress")
on_run_step(type="tool_calls", status="in_progress")      # if tools used
on_run_step_delta(...)                                     # streaming tool deltas
on_run_step(type="tool_calls", status="completed")
on_run_step(type="message_creation", status="in_progress")
on_message_delta(delta)                                    # token by token
on_message_delta(delta)
...
on_thread_message(status="completed")
on_run_step(type="message_creation", status="completed")
on_thread_run(status="completed")
on_done()
```

### Tool call retry

When the agent invokes a tool (`status == "requires_action"`), `AsyncAgentEventHandler._process_event` automatically calls `self.submit_tool_outputs()` — this is wired up by `AsyncToolSet`. If the tool output contains errors, the handler increments `current_retry` and retries up to `_max_retry` (default: 10).

To change the retry limit:

```python
handler = MyEventHandler()
handler.set_max_retry(3)  # Fail fast
```

### Example: Live token streaming with progress tracking

```python
import asyncio, os, sys
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import (
    AsyncAgentEventHandler,
    AsyncFunctionTool,
    AsyncToolSet,
    MessageDeltaChunk,
    ThreadMessage,
    ThreadRun,
    RunStep,
    RunStatus,
    MessageRole,
)
from azure.identity.aio import DefaultAzureCredential


class PrintingEventHandler(AsyncAgentEventHandler):
    """Print tokens as they arrive; track run steps for observability."""

    def __init__(self) -> None:
        super().__init__()
        self._step_count = 0
        self._token_count = 0

    async def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        for content in delta.delta.content or []:
            if hasattr(content, "text") and content.text:
                text = content.text.value or ""
                self._token_count += len(text.split())
                print(text, end="", flush=True)  # live streaming output

    async def on_run_step(self, step: RunStep) -> None:
        if step.status == "in_progress":
            self._step_count += 1
            print(f"\n[Step {self._step_count}: {step.type}]", flush=True)

    async def on_thread_run(self, run: ThreadRun) -> None:
        if run.status in (RunStatus.FAILED, RunStatus.EXPIRED):
            print(f"\n❌ Run {run.status}: {run.last_error}", file=sys.stderr)

    async def on_done(self) -> None:
        print(f"\n\n✓ Done — ~{self._token_count} words streamed across {self._step_count} steps")


async def get_current_temperature(city: str) -> str:
    """Get the current temperature for a city in Celsius.

    :param city: Name of the city.
    :type city: str
    :return: Temperature description.
    :rtype: str
    """
    # Stub — replace with real API call
    return f"The current temperature in {city} is 18°C."


async def main() -> None:
    tool = AsyncFunctionTool(functions={get_current_temperature})
    toolset = AsyncToolSet()
    toolset.add(tool)

    async with AgentsClient(
        endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
        credential=DefaultAzureCredential(),
    ) as client:
        agent = await client.create_agent(
            model="gpt-4o",
            name="streaming-agent",
            instructions="Answer questions. Use tools when you need live data.",
            tools=toolset.definitions,
        )
        thread = await client.threads.create()
        await client.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content=(
                "Compare the weather in London and Paris. "
                "Then write a two-paragraph travel recommendation."
            ),
        )

        async with await client.runs.stream(
            thread_id=thread.id,
            agent_id=agent.id,
            toolset=toolset,
            event_handler=PrintingEventHandler(),
        ) as stream:
            await stream.until_done()

        await client.threads.delete(thread.id)
        await client.delete_agent(agent.id)


asyncio.run(main())
```

### Example: Custom error handling and fallback

```python
import asyncio, os
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import (
    AsyncAgentEventHandler,
    ThreadRun,
    RunStatus,
    MessageRole,
)
from azure.identity.aio import DefaultAzureCredential


class ResilientEventHandler(AsyncAgentEventHandler):
    """Catches stream errors and stores them for the caller to handle."""

    def __init__(self) -> None:
        super().__init__()
        self.response_parts: list[str] = []
        self.errors: list[str] = []
        self.final_status: str | None = None

    async def on_message_delta(self, delta) -> None:
        for content in delta.delta.content or []:
            if hasattr(content, "text") and content.text:
                self.response_parts.append(content.text.value or "")

    async def on_thread_run(self, run: ThreadRun) -> None:
        self.final_status = run.status
        if run.status == RunStatus.FAILED and run.last_error:
            self.errors.append(
                f"Run failed (code={run.last_error.code}): {run.last_error.message}"
            )

    async def on_error(self, data: str) -> None:
        self.errors.append(f"Stream error: {data}")

    @property
    def full_response(self) -> str:
        return "".join(self.response_parts)


async def safe_stream(client, thread_id: str, agent_id: str) -> str:
    """Stream a run and return the full text, or an error message."""
    handler = ResilientEventHandler()
    handler.set_max_retry(2)  # limit tool retries

    try:
        async with await client.runs.stream(
            thread_id=thread_id,
            agent_id=agent_id,
            event_handler=handler,
        ) as stream:
            await stream.until_done()
    except Exception as exc:
        handler.errors.append(f"Unexpected stream exception: {exc}")

    if handler.errors:
        print(f"Errors during stream: {handler.errors}")

    return handler.full_response or "(no response)"
```

### Example: Collecting tool call metadata during streaming

```python
from azure.ai.agents.models import RunStep, RunStepType

class ObservingEventHandler(AsyncAgentEventHandler):
    """Collect tool call details for observability."""

    def __init__(self) -> None:
        super().__init__()
        self.tool_calls_made: list[dict] = []
        self._tokens_streamed = 0

    async def on_run_step(self, step: RunStep) -> None:
        if step.status == "completed" and step.type == RunStepType.TOOL_CALLS:
            for tc in step.step_details.tool_calls:
                if hasattr(tc, "function"):
                    self.tool_calls_made.append({
                        "name": tc.function.name,
                        "args": tc.function.arguments,
                        "result": tc.function.output,
                    })

    async def on_message_delta(self, delta) -> None:
        for content in delta.delta.content or []:
            if hasattr(content, "text") and content.text:
                self._tokens_streamed += 1

    async def on_done(self) -> None:
        print(f"Tool calls: {len(self.tool_calls_made)}")
        print(f"Tokens streamed: {self._tokens_streamed}")
        for tc in self.tool_calls_made:
            print(f"  {tc['name']}({tc['args']}) → {tc['result'][:80]}...")
```

### `AsyncAgentEventHandler` vs `AgentEventHandler`

| | `AgentEventHandler` | `AsyncAgentEventHandler` |
|---|---|---|
| `on_*` methods | `def` (sync) | `async def` (coroutines) |
| Client | Sync `AgentsClient` | Async `AgentsClient` (`aio`) |
| Tool functions | Only sync callables | Sync or async callables via `AsyncToolSet` |
| `execute()` calls | Sync dispatch | Awaited dispatch |
| Use with | `client.runs.stream(...)` | `await client.runs.stream(...)` |

---

## Patterns Combining Multiple Classes

### Pattern 1: Async agent with structured output and streaming

Combines `AsyncFunctionTool`, `AsyncAgentEventHandler`, and `ResponseFormatJsonSchemaType` for a fully async, streaming, structured-output agent.

```python
import asyncio, json, os
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import (
    AsyncFunctionTool,
    AsyncToolSet,
    AsyncAgentEventHandler,
    MessageDeltaChunk,
    ResponseFormatJsonSchema,
    ResponseFormatJsonSchemaType,
    MessageRole,
)
from azure.identity.aio import DefaultAzureCredential


async def lookup_product(sku: str) -> str:
    """Look up product details by SKU.

    :param sku: The product SKU code.
    :type sku: str
    :return: Product details as a JSON string.
    :rtype: str
    """
    products = {
        "BOOT-42": {"name": "Trail Boot", "price": 89.99, "stock": 14},
        "JACK-L": {"name": "Waterproof Jacket", "price": 124.50, "stock": 3},
    }
    return json.dumps(products.get(sku, {"error": "SKU not found"}))


PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "sku": {"type": "string"},
        "product_name": {"type": "string"},
        "price_gbp": {"type": "number"},
        "stock_available": {"type": "integer"},
        "recommendation": {"type": "string"},
    },
    "required": ["sku", "product_name", "price_gbp", "stock_available", "recommendation"],
    "additionalProperties": False,
}


class StreamHandler(AsyncAgentEventHandler):
    async def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        for content in delta.delta.content or []:
            if hasattr(content, "text") and content.text:
                print(content.text.value or "", end="", flush=True)

    async def on_done(self) -> None:
        print()  # newline after stream ends


async def main() -> None:
    tool = AsyncFunctionTool(functions={lookup_product})
    toolset = AsyncToolSet()
    toolset.add(tool)

    async with AgentsClient(
        endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
        credential=DefaultAzureCredential(),
    ) as client:
        agent = await client.create_agent(
            model="gpt-4o",
            name="product-agent",
            instructions=(
                "You are a retail assistant. Look up the product and return "
                "structured information including a stock recommendation."
            ),
            tools=toolset.definitions,
            response_format=ResponseFormatJsonSchemaType(
                json_schema=ResponseFormatJsonSchema(
                    name="product_detail",
                    schema=PRODUCT_SCHEMA,
                )
            ),
        )

        thread = await client.threads.create()
        await client.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content="Tell me about SKU BOOT-42",
        )

        async with await client.runs.stream(
            thread_id=thread.id,
            agent_id=agent.id,
            toolset=toolset,
            event_handler=StreamHandler(),
        ) as stream:
            await stream.until_done()

        await client.threads.delete(thread.id)
        await client.delete_agent(agent.id)


asyncio.run(main())
```

### Pattern 2: Run introspection dashboard

Combines `ThreadRun`, `RunStep`, and `RunStepType` to build a cost/audit summary after any run.

```python
def run_dashboard(client, thread_id: str, run_id: str) -> None:
    """Print a summary dashboard for a completed run."""
    from azure.ai.agents.models import RunStatus, RunStepType

    run = client.runs.get(thread_id=thread_id, run_id=run_id)

    print("=" * 60)
    print(f"Run: {run.id}")
    print(f"Status: {run.status}")
    print(f"Model: {run.model}")
    if run.started_at and run.completed_at:
        elapsed = (run.completed_at - run.started_at).total_seconds()
        print(f"Elapsed: {elapsed:.1f}s")
    if run.usage:
        print(f"Tokens — prompt: {run.usage.prompt_tokens:,}, "
              f"completion: {run.usage.completion_tokens:,}, "
              f"total: {run.usage.total_tokens:,}")
    print()

    steps = list(client.run_steps.list(thread_id=thread_id, run_id=run_id))
    print(f"Steps: {len(steps)}")
    for step in sorted(steps, key=lambda s: s.created_at):
        status_icon = "✓" if step.status == "completed" else "✗"
        print(f"  {status_icon} [{step.type}] {step.status}", end="")
        if step.usage:
            print(f" ({step.usage.total_tokens} tokens)", end="")
        print()

        if step.type == RunStepType.TOOL_CALLS:
            for tc in step.step_details.tool_calls:
                name = getattr(getattr(tc, "function", None), "name", tc.type)
                print(f"      ↳ {name}")
    print("=" * 60)
```

### Pattern 3: Managed knowledge base with vector store lifecycle

Combines `VectorStore`, `VectorStoreExpirationPolicy`, `AzureAISearchTool` and `FileSearchTool` for a managed hybrid search agent.

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    AzureAISearchTool, AzureAISearchQueryType,
    FileSearchTool,
    VectorStoreExpirationPolicy,
    VectorStoreExpirationPolicyAnchor,
    FilePurpose,
    MessageRole,
    ToolSet,
)
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# Upload a local document
with open("internal-policy.pdf", "rb") as f:
    file_obj = client.files.upload(file=f, purpose=FilePurpose.AGENTS)

# Create a vector store for the uploaded document
store = client.vector_stores.create_and_poll(
    file_ids=[file_obj.id],
    name="internal-policies",
    expires_after=VectorStoreExpirationPolicy(
        anchor=VectorStoreExpirationPolicyAnchor.LAST_ACTIVE_AT,
        days=14,
    ),
)
print(f"Vector store ready: {store.id} ({store.file_counts.completed} files)")

# Azure AI Search for external knowledge
azure_search = AzureAISearchTool(
    index_connection_id=os.environ["AZURE_SEARCH_CONNECTION_ID"],
    index_name="industry-regulations",
    query_type=AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
    top_k=5,
)

# FileSearch for uploaded documents
file_search = FileSearchTool(vector_store_ids=[store.id])

toolset = ToolSet()
toolset.add(azure_search)
toolset.add(file_search)

agent = client.create_agent(
    model="gpt-4o",
    name="compliance-agent",
    instructions=(
        "You are a compliance officer. Use internal policies (file_search) "
        "and industry regulations (azure_ai_search) to answer questions accurately."
    ),
    tools=toolset.definitions,
    tool_resources=toolset.resources,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="Does our internal data retention policy comply with the latest GDPR requirements?",
)
run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(tc.text.value)

# Cleanup
client.threads.delete(thread.id)
client.delete_agent(agent.id)
client.vector_stores.delete(store.id)
client.files.delete(file_obj.id)
client.close()
```

---

## Quick Reference Table

| Class | Import from | When to use |
|---|---|---|
| `AsyncFunctionTool` | `azure.ai.agents.models` | Async tool functions; mix of sync + async callables |
| `AzureFunctionTool` + `AzureFunctionStorageQueue` | `azure.ai.agents.models` | Azure Functions triggered via Storage Queues |
| `AzureAISearchTool` | `azure.ai.agents.models` | Existing Azure AI Search indexes as agent knowledge |
| `VectorStore` + ops | `azure.ai.agents.models` | Manage and monitor server-side vector stores |
| `ThreadRun` + `RunStatus` | `azure.ai.agents.models` | Inspect and control agent run execution |
| `RunStep` + step detail subtypes | `azure.ai.agents.models` | Audit individual tool calls and message creation |
| `ResponseFormatJsonSchema` + `ResponseFormatJsonSchemaType` | `azure.ai.agents.models` | Constrain agent output to a JSON schema |
| `TruncationObject` + `TruncationStrategy` | `azure.ai.agents.models` | Control which thread history goes into context |
| `MessageAttachment` | `azure.ai.agents.models` | Attach per-message files for FileSearch or CodeInterpreter |
| `AsyncAgentEventHandler` | `azure.ai.agents.models` | Custom async streaming with token-level callbacks |

See also: [Class Deep Dives Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sdk_class_deep_dives/) for `AgentsClient`, `FunctionTool`, `ToolSet`, `CodeInterpreterTool`, `FileSearchTool`, `BingGroundingTool`, `ConnectedAgentTool`, `AgentEventHandler`, `ThreadMessage`, and `OpenApiTool`.

---
title: "Azure AI Agents SDK (Python) — Class Deep Dives Vol. 4"
description: "Source-verified deep dives into 10 classes from azure-ai-agents 1.1.0: AgentsClient (enable_auto_function_calls, create_thread_and_process_run), FunctionTool, ToolSet, CodeInterpreterTool with file upload, FileSearchTool with VectorStore lifecycle, AzureAISearchTool query modes, BingGroundingTool parameters, ConnectedAgentTool multi-agent orchestration, AgentEventHandler custom streaming subclass, and AsyncToolSet with AsyncFunctionTool."
framework: microsoft-agent-framework
language: python
---

# Azure AI Agents SDK (Python) — Class Deep Dives Vol. 4

**Package:** `azure-ai-agents`  
**Version covered:** 1.1.0  
**Verified against:** installed package at `/usr/local/lib/python3.11/dist-packages/azure/ai/agents/`

This is the fourth volume of source-verified class deep dives for the `azure-ai-agents` Python SDK. Each section includes the real class signature derived from the installed source, followed by practical, runnable code examples. This volume emphasises **patterns and combinations** that are not covered in detail in earlier volumes — specifically tool composition, file operations, multi-agent orchestration, and custom streaming.

Earlier volumes:
- **[Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sdk_class_deep_dives/)** — `AgentsClient`, `FunctionTool`, `ToolSet`, `CodeInterpreterTool`, `FileSearchTool`, `BingGroundingTool`, `ConnectedAgentTool`, `AgentEventHandler`, `ThreadMessage`, `OpenApiTool`
- **[Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sdk_class_deep_dives_v3/)** — `AsyncFunctionTool`, `AzureFunctionTool`, `AzureAISearchTool`, `VectorStore`, `ThreadRun`, `RunStep`, `ResponseFormatJsonSchema`, `TruncationObject`, `MessageAttachment`, `AsyncAgentEventHandler`

---

## Table of Contents

1. [AgentsClient — enable_auto_function_calls and create_thread_and_process_run](#1-agentsclient--enable_auto_function_calls-and-create_thread_and_process_run)
2. [FunctionTool — error handling, schema introspection, and dynamic registration](#2-functiontool--error-handling-schema-introspection-and-dynamic-registration)
3. [ToolSet — composing multiple tools with type validation](#3-toolset--composing-multiple-tools-with-type-validation)
4. [CodeInterpreterTool — file upload, execution, and run step inspection](#4-codeinterpretertool--file-upload-execution-and-run-step-inspection)
5. [FileSearchTool + VectorStore — full lifecycle with expiry policy](#5-filesearchtool--vectorstore--full-lifecycle-with-expiry-policy)
6. [AzureAISearchTool — simple, semantic, and hybrid query modes](#6-azureaisearchtool--simple-semantic-and-hybrid-query-modes)
7. [BingGroundingTool — market, count, freshness, and language parameters](#7-binggroundingtool--market-count-freshness-and-language-parameters)
8. [ConnectedAgentTool — multi-agent orchestration end to end](#8-connectedagenttool--multi-agent-orchestration-end-to-end)
9. [AgentEventHandler — custom streaming subclass](#9-agenteventhandler--custom-streaming-subclass)
10. [AsyncToolSet + AsyncFunctionTool — concurrent async tool execution](#10-asynctoolset--asyncfunctiontool--concurrent-async-tool-execution)

---

## 1. `AgentsClient` — `enable_auto_function_calls` and `create_thread_and_process_run`

Vol. 1 covered `AgentsClient` construction and the basic run lifecycle. This section digs into two convenience APIs that significantly reduce boilerplate in production usage.

### `enable_auto_function_calls`

**Source:** `azure/ai/agents/_patch.py`

```python
def enable_auto_function_calls(
    self,
    tools: Union[Set[Callable[..., Any]], FunctionTool, ToolSet],
    max_retry: int = 10,
) -> None:
```

Registers functions on the client so that when a run reaches `requires_action`, the SDK automatically executes the requested function calls and resubmits the outputs — no polling loop required on your side.

**Why this matters:** Without `enable_auto_function_calls` you have to write a polling loop that checks `run.status == "requires_action"`, dispatches each `RequiredFunctionToolCall`, collects outputs, calls `runs.submit_tool_outputs`, and then re-polls. `enable_auto_function_calls` + `runs.create_and_process` or `runs.stream` handles all of that internally.

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import FunctionTool, ToolSet, MessageRole
from azure.identity import DefaultAzureCredential

endpoint = os.environ["AZURE_AI_AGENTS_ENDPOINT"]

# ── Define callable functions ────────────────────────────────────────────────
def get_weather(city: str) -> str:
    """Return current weather for the given city."""
    # Replace with a real API call in production
    return f"{city}: 18°C, partly cloudy"


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between two currencies."""
    rates = {"USD_EUR": 0.92, "EUR_USD": 1.09}
    key = f"{from_currency}_{to_currency}"
    rate = rates.get(key, 1.0)
    result = round(amount * rate, 2)
    return f"{amount} {from_currency} = {result} {to_currency}"


# ── Build the client and register auto function calls ─────────────────────────
client = AgentsClient(endpoint=endpoint, credential=DefaultAzureCredential())

# Pass a plain Python set of callables — the SDK wraps them in a FunctionTool internally.
client.enable_auto_function_calls(
    tools={get_weather, convert_currency},
    max_retry=5,    # allow up to 5 retries when tool outputs contain errors
)

# ── Create an agent that declares the same functions ─────────────────────────
tool = FunctionTool({get_weather, convert_currency})

agent = client.create_agent(
    model="gpt-4o",
    name="multi-tool-agent",
    instructions="You are a helpful assistant. Use available tools to answer questions.",
    tools=tool.definitions,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="What's the weather in Berlin? And convert 100 USD to EUR.",
)

# create_and_process polls until done AND handles requires_action automatically.
run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
print(f"Run status: {run.status}")

# Read response
for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for text in msg.text_messages:
            print(text.text.value)

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

**Passing a `ToolSet` instead of a set of callables:**

```python
from azure.ai.agents.models import ToolSet, FunctionTool, CodeInterpreterTool

toolset = ToolSet()
toolset.add(FunctionTool({get_weather, convert_currency}))
toolset.add(CodeInterpreterTool())

# enable_auto_function_calls extracts the FunctionTool from the ToolSet automatically.
client.enable_auto_function_calls(tools=toolset)

# When creating the agent, pass the full toolset definitions and resources.
agent = client.create_agent(
    model="gpt-4o",
    name="toolset-agent",
    instructions="You can run Python code and look up weather.",
    toolset=toolset,
)
```

---

### `create_thread_and_process_run`

**Source:** `azure/ai/agents/_patch.py`

```python
def create_thread_and_process_run(
    self,
    *,
    agent_id: str,
    thread: Optional[AgentThreadCreationOptions] = None,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
    toolset: Optional[ToolSet] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_prompt_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    truncation_strategy: Optional[TruncationObject] = None,
    tool_choice: Optional[AgentsToolChoiceOption] = None,
    response_format: Optional[AgentsResponseFormatOption] = None,
    parallel_tool_calls: Optional[bool] = None,
    metadata: Optional[Dict[str, str]] = None,
    polling_interval: int = 1,
    **kwargs: Any,
) -> ThreadRun:
```

Creates a thread **and** runs the agent in a single call, then polls until completion. It combines `threads.create()`, `messages.create()` (via the initial message in `thread`), `runs.create()`, and polling — plus automatic tool-call dispatch when `enable_auto_function_calls` has been called.

```python
from azure.ai.agents.models import AgentThreadCreationOptions, ThreadMessageOptions, MessageRole

# Pre-load initial messages into the thread creation options
thread_options = AgentThreadCreationOptions(
    messages=[
        ThreadMessageOptions(
            role=MessageRole.USER,
            content="Analyse the attached CSV and summarise the key trends.",
        )
    ]
)

run = client.create_thread_and_process_run(
    agent_id=agent.id,
    thread=thread_options,
    temperature=0.3,
    max_completion_tokens=2048,
    metadata={"job_id": "batch-42", "customer": "acme"},
)

print(f"Thread: {run.thread_id}")
print(f"Status: {run.status}")           # "completed", "failed", "cancelled", etc.
print(f"Tokens used: {run.usage.total_tokens if run.usage else 'N/A'}")
```

**Override model and instructions per-run** (useful for A/B testing):

```python
run = client.create_thread_and_process_run(
    agent_id=agent.id,
    thread=AgentThreadCreationOptions(
        messages=[ThreadMessageOptions(role=MessageRole.USER, content="Hello")]
    ),
    model="gpt-4o-mini",       # override the agent's registered model
    instructions="Be very concise — one sentence only.",
    temperature=0.0,
)
```

---

## 2. `FunctionTool` — error handling, schema introspection, and dynamic registration

**Source:** `azure/ai/agents/models/_patch.py` — `FunctionTool(BaseFunctionTool)`

```python
class FunctionTool(BaseFunctionTool):
    def execute(self, tool_call: RequiredFunctionToolCall) -> Any:
        try:
            function, parsed_arguments = self._get_func_and_args(tool_call)
            return function(**parsed_arguments) if parsed_arguments else function()
        except Exception as e:
            error_message = f"Error executing function '{tool_call.function.name}': {e}"
            logger.error(error_message)
            return json.dumps({"error": error_message})
```

Key behaviour to internalise:

- **Exceptions are caught and returned as JSON** — the agent receives the error string and can adjust its approach (self-correction). This is intentional; do not raise inside tool functions unless you want an unhandled exception to propagate.
- The schema is derived from the **function's type annotations and docstring**. Arguments not annotated default to `string`.
- **`add(func)`** registers additional callables after construction.

### Minimal example with docstring-driven schema

```python
from azure.ai.agents.models import FunctionTool

def search_products(query: str, max_results: int = 10, in_stock_only: bool = False) -> str:
    """
    Search the product catalogue.

    :param query: The search term.
    :param max_results: Maximum number of results to return (1-100).
    :param in_stock_only: If true, only return products currently in stock.
    :return: JSON array of matching products.
    """
    # Simulate a DB query
    results = [{"id": 1, "name": "Widget A", "in_stock": True}]
    if in_stock_only:
        results = [r for r in results if r["in_stock"]]
    return str(results[:max_results])

tool = FunctionTool({search_products})

# Inspect what the SDK will send to the model
for defn in tool.definitions:
    print(defn.type)             # "function"
    print(defn.function.name)    # "search_products"
    print(defn.function.description)
    print(defn.function.parameters)   # JSON schema derived from annotations
```

### Dynamic registration with `add_functions()`

```python
import json

def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Fetch the current exchange rate between two currencies."""
    # Stub — replace with a real rates API
    return json.dumps({"rate": 0.92, "source": "ECB"})

tool = FunctionTool(set())                          # start with no functions
tool.add_functions({search_products})               # register at runtime
tool.add_functions({get_exchange_rate})             # add a second function

print(f"Registered {len(tool.definitions)} tools")
```

### Handling errors gracefully inside tools

Because `FunctionTool.execute` catches all exceptions and returns them as JSON error strings, the model receives actionable feedback:

```python
def divide(numerator: float, denominator: float) -> str:
    """Divide numerator by denominator."""
    if denominator == 0:
        raise ValueError("Denominator cannot be zero — provide a non-zero value.")
    return str(numerator / denominator)

# The agent will see: {"error": "Error executing function 'divide': Denominator cannot be zero — ..."}
# and can ask the user for a different denominator or rephrase its answer.
```

---

## 3. `ToolSet` — composing multiple tools with type validation

**Source:** `azure/ai/agents/models/_patch.py` — `ToolSet(BaseToolSet)`

```python
class ToolSet(BaseToolSet):
    def validate_tool_type(self, tool: Tool) -> None:
        if isinstance(tool, AsyncFunctionTool):
            raise ValueError(
                "AsyncFunctionTool is not supported in ToolSet. "
                "To use async functions, use AsyncToolSet and agents operations in azure.ai.agents.aio."
            )

    def execute_tool_calls(self, tool_calls: List[Any]) -> Any: ...
```

`ToolSet` is the synchronous container for all tool types. It enforces that you don't accidentally mix `AsyncFunctionTool` with the sync client.

### Composing built-in tools

```python
from azure.ai.agents.models import (
    ToolSet, FunctionTool, CodeInterpreterTool, FileSearchTool,
)

def lookup_order(order_id: str) -> str:
    """Look up order status by order ID."""
    return f"Order {order_id}: shipped on 2026-05-20"

toolset = ToolSet()
toolset.add(FunctionTool({lookup_order}))   # custom function
toolset.add(CodeInterpreterTool())           # built-in code execution
toolset.add(FileSearchTool(["vs_abc123"]))  # built-in file search

agent = client.create_agent(
    model="gpt-4o",
    name="multi-capability-agent",
    instructions="You can look up orders, run code, and search documents.",
    toolset=toolset,         # pass the entire ToolSet — definitions AND resources together
)
```

### Retrieving a specific tool from the set

```python
# Useful when you need to modify a tool after construction
function_tool = toolset.get_tool(FunctionTool)
function_tool.add_functions({get_exchange_rate})   # add another function dynamically
```

### `execute_tool_calls` — what happens internally

`ToolSet.execute_tool_calls` is called by the SDK's run-polling loop when a run enters `requires_action`. You normally never call this directly, but understanding it helps with debugging:

```python
# Internally the SDK does something like:
# required_action = run.required_action
# tool_calls = required_action.submit_tool_outputs.tool_calls
# outputs = toolset.execute_tool_calls(tool_calls)
# client.runs.submit_tool_outputs(thread_id=..., run_id=..., tool_outputs=outputs)
```

Each output is a dict `{"tool_call_id": str, "output": str}`. The SDK submits them all in one call.

---

## 4. `CodeInterpreterTool` — file upload, execution, and run step inspection

**Source:** `azure/ai/agents/models/_patch.py` — `CodeInterpreterTool(Tool[CodeInterpreterToolDefinition])`

```python
class CodeInterpreterTool(Tool[CodeInterpreterToolDefinition]):
    def __init__(self, file_ids: Optional[List[str]] = None): ...
    def add_file(self, file_id: str) -> None: ...
    def remove_file(self, file_id: str) -> None: ...

    @property
    def definitions(self) -> List[CodeInterpreterToolDefinition]: ...
    @property
    def resources(self) -> ToolResources: ...   # wraps file_ids in CodeInterpreterToolResource
```

The Code Interpreter tool lets the agent write and execute Python code in a sandboxed environment, including reading files you upload.

### Full example: upload a file, run analysis, inspect run steps

```python
import os
import json
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    CodeInterpreterTool, FilePurpose, MessageRole,
    RunStepCodeInterpreterToolCall,
)
from azure.identity import DefaultAzureCredential

endpoint = os.environ["AZURE_AI_AGENTS_ENDPOINT"]
client = AgentsClient(endpoint=endpoint, credential=DefaultAzureCredential())

# ── 1. Upload a CSV file for the agent to analyse ────────────────────────────
with open("sales.csv", "w") as f:
    f.write("month,revenue\nJanuary,12000\nFebruary,15500\nMarch,9800\n")

file_info = client.files.upload(
    file_path="sales.csv",
    purpose=FilePurpose.ASSISTANTS,
)
print(f"Uploaded file: {file_info.id}")

# ── 2. Build the tool with the uploaded file attached ────────────────────────
code_tool = CodeInterpreterTool(file_ids=[file_info.id])

# ── 3. Create an agent ───────────────────────────────────────────────────────
agent = client.create_agent(
    model="gpt-4o",
    name="data-analyst",
    instructions=(
        "You are a data analyst. Use the code interpreter to analyse uploaded CSV files "
        "and return clear summaries with key statistics."
    ),
    tools=code_tool.definitions,
    tool_resources=code_tool.resources,
)

# ── 4. Create thread and post the user question ──────────────────────────────
thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="Analyse the sales data and identify the best and worst performing months.",
)

# ── 5. Run and poll ──────────────────────────────────────────────────────────
run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
print(f"Run status: {run.status}")

# ── 6. Inspect run steps to see what code was executed ───────────────────────
steps = client.run_steps.list(thread_id=thread.id, run_id=run.id)
for step in steps:
    print(f"Step type: {step.type}, status: {step.status}")
    if step.type == "tool_calls":
        for tool_call in step.step_details.tool_calls:
            if isinstance(tool_call, RunStepCodeInterpreterToolCall):
                ci = tool_call.code_interpreter
                print("─── Code executed ───")
                print(ci.input)
                print("─── Output ──────────")
                for output in ci.outputs:
                    if hasattr(output, "logs"):
                        print(output.logs)

# ── 7. Read the response ─────────────────────────────────────────────────────
for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for content in msg.text_messages:
            print(content.text.value)

# ── 8. Cleanup ────────────────────────────────────────────────────────────────
client.threads.delete(thread.id)
client.files.delete(file_info.id)
client.delete_agent(agent.id)
```

### Adding files after agent creation

```python
# You can upload additional files and register them on the fly
new_file = client.files.upload(file_path="q2_sales.csv", purpose=FilePurpose.ASSISTANTS)
code_tool.add_file(new_file.id)

# Then update the agent with the new resources
client.update_agent(
    agent_id=agent.id,
    tool_resources=code_tool.resources,
)
```

---

## 5. `FileSearchTool` + `VectorStore` — full lifecycle with expiry policy

**Source:** `azure/ai/agents/models/_patch.py` — `FileSearchTool(Tool[FileSearchToolDefinition])`

```python
class FileSearchTool(Tool[FileSearchToolDefinition]):
    def __init__(self, vector_store_ids: Optional[List[str]] = None): ...
    def add_vector_store(self, store_id: str) -> None: ...
    def remove_vector_store(self, store_id: str) -> None: ...

    @property
    def definitions(self) -> List[FileSearchToolDefinition]: ...
    @property
    def resources(self) -> ToolResources: ...
```

`FileSearchTool` allows the agent to semantically search documents stored in a vector store. The vector store is managed separately via `client.vector_stores`.

### End-to-end: create vector store, upload files, run file search

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    FileSearchTool, FilePurpose, MessageRole,
    VectorStoreExpirationPolicy, VectorStoreExpirationPolicyAnchor,
)
from azure.identity import DefaultAzureCredential

endpoint = os.environ["AZURE_AI_AGENTS_ENDPOINT"]
client = AgentsClient(endpoint=endpoint, credential=DefaultAzureCredential())

# ── 1. Upload documents ──────────────────────────────────────────────────────
with open("handbook.txt", "w") as f:
    f.write("Chapter 1: Onboarding\nAll new employees must complete HR orientation.\n")

file_info = client.files.upload(
    file_path="handbook.txt",
    purpose=FilePurpose.ASSISTANTS,
)

# ── 2. Create a vector store and poll until ready ────────────────────────────
# VectorStoreExpirationPolicy auto-expires the store 7 days after last use.
expiry_policy = VectorStoreExpirationPolicy(
    anchor=VectorStoreExpirationPolicyAnchor.LAST_ACTIVE_AT,
    days=7,
)

vector_store = client.vector_stores.create_and_poll(
    name="employee-handbook",
    file_ids=[file_info.id],
    expires_after=expiry_policy,
    metadata={"version": "2026-Q2", "owner": "hr-team"},
)
print(f"Vector store ready: {vector_store.id}, status: {vector_store.status}")
print(f"File count: {vector_store.file_counts.completed} completed")

# ── 3. Create file search tool pointing at the vector store ──────────────────
file_tool = FileSearchTool(vector_store_ids=[vector_store.id])

# ── 4. Create agent ──────────────────────────────────────────────────────────
agent = client.create_agent(
    model="gpt-4o",
    name="hr-assistant",
    instructions="You answer HR questions using the uploaded handbook. Cite sections where relevant.",
    tools=file_tool.definitions,
    tool_resources=file_tool.resources,
)

# ── 5. Ask a question ────────────────────────────────────────────────────────
thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="What does the onboarding process require of new employees?",
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for content in msg.text_messages:
            print(content.text.value)
            # File citations appear in msg.text_messages[n].text.annotations
            for annotation in content.text.annotations:
                print(f"  Citation: {annotation}")

# ── 6. Cleanup ────────────────────────────────────────────────────────────────
client.threads.delete(thread.id)
client.vector_stores.delete(vector_store.id)
client.files.delete(file_info.id)
client.delete_agent(agent.id)
```

### Adding a second vector store after the agent is created

```python
second_store = client.vector_stores.create_and_poll(
    name="legal-policies",
    file_ids=[legal_file_id],
)

# Register the new store on the existing tool
file_tool.add_vector_store(second_store.id)

# Update the agent to use both vector stores
client.update_agent(
    agent_id=agent.id,
    tool_resources=file_tool.resources,
)
```

---

## 6. `AzureAISearchTool` — simple, semantic, and hybrid query modes

**Source:** `azure/ai/agents/models/_patch.py` — `AzureAISearchTool(Tool[AzureAISearchToolDefinition])`

```python
class AzureAISearchTool(Tool[AzureAISearchToolDefinition]):
    def __init__(
        self,
        index_connection_id: str,
        index_name: str,
        query_type: AzureAISearchQueryType = AzureAISearchQueryType.SIMPLE,
        filter: str = "",
        top_k: int = 5,
        index_asset_id: str = "",
    ): ...
```

`AzureAISearchTool` connects an agent to an Azure AI Search index. The `query_type` parameter governs how the search query is formed — this is the most important tuning knob.

### Query type comparison

| `query_type` | When to use |
|---|---|
| `SIMPLE` | Keyword search. Fast, no semantic model required. Use for structured queries, SKU lookups, or exact phrase matching. |
| `SEMANTIC` | Semantic re-ranking with a language model. Higher relevance for natural-language questions. Requires a semantic configuration in the index. |
| `VECTOR` | Pure vector similarity search. Best when your index stores embedding vectors. |
| `VECTOR_SIMPLE_HYBRID` | Combines keyword + vector search. Good general-purpose choice when your index has both full-text and vectors. |
| `VECTOR_SEMANTIC_HYBRID` | Combines vector + semantic re-ranking. Highest quality, but also highest latency and cost. |

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import AzureAISearchTool, AzureAISearchQueryType, MessageRole
from azure.identity import DefaultAzureCredential

endpoint = os.environ["AZURE_AI_AGENTS_ENDPOINT"]
client = AgentsClient(endpoint=endpoint, credential=DefaultAzureCredential())

CONNECTION_ID = os.environ["AZURE_AI_SEARCH_CONNECTION_ID"]   # Connection resource ID
INDEX_NAME = "product-catalogue"

# ── Simple keyword search (default) ─────────────────────────────────────────
simple_tool = AzureAISearchTool(
    index_connection_id=CONNECTION_ID,
    index_name=INDEX_NAME,
    query_type=AzureAISearchQueryType.SIMPLE,
    top_k=10,
)

# ── Semantic search ──────────────────────────────────────────────────────────
semantic_tool = AzureAISearchTool(
    index_connection_id=CONNECTION_ID,
    index_name=INDEX_NAME,
    query_type=AzureAISearchQueryType.SEMANTIC,
    top_k=5,
)

# ── Hybrid vector + semantic (best quality) ──────────────────────────────────
hybrid_tool = AzureAISearchTool(
    index_connection_id=CONNECTION_ID,
    index_name=INDEX_NAME,
    query_type=AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
    top_k=5,
    filter="category eq 'electronics'",   # OData filter expression
)

# ── Create an agent with hybrid search ──────────────────────────────────────
agent = client.create_agent(
    model="gpt-4o",
    name="product-search-agent",
    instructions=(
        "You help customers find products using the Azure AI Search index. "
        "Always cite the product name and category in your response."
    ),
    tools=hybrid_tool.definitions,
    tool_resources=hybrid_tool.resources,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="Find me a wireless noise-cancelling headset under £150.",
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for content in msg.text_messages:
            print(content.text.value)

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

---

## 7. `BingGroundingTool` — market, count, freshness, and language parameters

**Source:** `azure/ai/agents/models/_patch.py` — `BingGroundingTool(Tool[BingGroundingToolDefinition])`

```python
class BingGroundingTool(Tool[BingGroundingToolDefinition]):
    def __init__(
        self,
        connection_id: str,
        market: str = "",
        set_lang: str = "",
        count: int = 5,
        freshness: str = "",
    ): ...
```

`BingGroundingTool` gives the agent access to real-time web search via Bing. Its `__init__` parameters map directly to Bing Search API request parameters.

| Parameter | Type | Effect |
|-----------|------|--------|
| `connection_id` | `str` | The Azure connection resource ID for your Bing Custom Search or Bing Search resource |
| `market` | `str` | Bing market code, e.g. `"en-GB"`, `"de-DE"`, `"fr-FR"`. Controls the search region and language of results |
| `set_lang` | `str` | UI language for result labels (distinct from `market`). E.g. `"en"`, `"de"` |
| `count` | `int` | Number of results Bing returns. Default is 5; maximum is typically 50 |
| `freshness` | `str` | Filter results by age: `"Day"` (past 24 h), `"Week"` (past 7 days), `"Month"` (past 30 days), or an exact date range like `"2025-01-01..2025-12-31"` |

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import BingGroundingTool, MessageRole
from azure.identity import DefaultAzureCredential

endpoint = os.environ["AZURE_AI_AGENTS_ENDPOINT"]
BING_CONNECTION_ID = os.environ["BING_CONNECTION_ID"]

client = AgentsClient(endpoint=endpoint, credential=DefaultAzureCredential())

# ── News-focused agent: UK market, recent results only ───────────────────────
bing_tool = BingGroundingTool(
    connection_id=BING_CONNECTION_ID,
    market="en-GB",           # UK search results
    set_lang="en",            # English UI labels
    count=10,                 # retrieve up to 10 results
    freshness="Week",         # only results from the past 7 days
)

agent = client.create_agent(
    model="gpt-4o",
    name="news-agent",
    instructions=(
        "You are a news analyst specialising in UK financial markets. "
        "Summarise news using Bing and highlight market-moving events."
    ),
    tools=bing_tool.definitions,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="What are the key UK market stories from the past week?",
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for content in msg.text_messages:
            print(content.text.value)

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

### Multiple regional agents from one pattern

```python
REGIONS = [
    {"market": "en-US", "lang": "en", "name": "us-news-agent"},
    {"market": "de-DE", "lang": "de", "name": "de-news-agent"},
    {"market": "ja-JP", "lang": "ja", "name": "jp-news-agent"},
]

agents = []
for region in REGIONS:
    tool = BingGroundingTool(
        connection_id=BING_CONNECTION_ID,
        market=region["market"],
        set_lang=region["lang"],
        count=5,
        freshness="Day",
    )
    ag = client.create_agent(
        model="gpt-4o",
        name=region["name"],
        instructions=f"You summarise {region['market']} news in the native language.",
        tools=tool.definitions,
    )
    agents.append(ag)
```

---

## 8. `ConnectedAgentTool` — multi-agent orchestration end to end

**Source:** `azure/ai/agents/models/_patch.py` — `ConnectedAgentTool(Tool[ConnectedAgentToolDefinition])`

```python
class ConnectedAgentTool(Tool[ConnectedAgentToolDefinition]):
    def __init__(self, id: str, name: str, description: str): ...
```

`ConnectedAgentTool` is how you wire one agent to call another. The `description` parameter is critical — it tells the **orchestrator agent** when and why to delegate to the connected (sub) agent.

### Architecture

```
User → OrchestratorAgent
         ├── ConnectedAgentTool("billing-agent", ...)  →  BillingAgent
         └── ConnectedAgentTool("logistics-agent", ...) →  LogisticsAgent
```

The orchestrator agent sees the connected agents as black-box tools. When the orchestrator decides to call one, the Azure AI Agents service routes the call to the connected agent and returns its response.

### Complete example

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import ConnectedAgentTool, FunctionTool, MessageRole
from azure.identity import DefaultAzureCredential

endpoint = os.environ["AZURE_AI_AGENTS_ENDPOINT"]
client = AgentsClient(endpoint=endpoint, credential=DefaultAzureCredential())

# ── 1. Create specialist agents ───────────────────────────────────────────────

def get_invoice(invoice_id: str) -> str:
    """Retrieve an invoice by ID."""
    return f"Invoice {invoice_id}: £2,400 due 2026-06-01, status: unpaid"

def process_refund(order_id: str, amount: float, reason: str) -> str:
    """Process a customer refund."""
    return f"Refund of £{amount:.2f} for order {order_id} approved. Reason: {reason}."

billing_tools = FunctionTool({get_invoice, process_refund})
billing_agent = client.create_agent(
    model="gpt-4o",
    name="billing-agent",
    instructions=(
        "You handle billing enquiries: invoice lookups, payment status, and refund processing. "
        "Always confirm refund amounts before processing."
    ),
    tools=billing_tools.definitions,
)

def track_shipment(tracking_number: str) -> str:
    """Track a shipment by its tracking number."""
    return f"Tracking {tracking_number}: In transit, expected delivery 2026-06-03"

def get_delivery_estimate(postcode: str) -> str:
    """Get the standard delivery estimate for a postcode."""
    return f"Standard delivery to {postcode}: 2-3 business days"

logistics_tools = FunctionTool({track_shipment, get_delivery_estimate})
logistics_agent = client.create_agent(
    model="gpt-4o",
    name="logistics-agent",
    instructions=(
        "You handle order tracking and delivery enquiries. "
        "Provide clear estimates and tracking updates."
    ),
    tools=logistics_tools.definitions,
)

# ── 2. Create orchestrator with connected agent tools ─────────────────────────
# The description drives when the orchestrator routes to each sub-agent.
billing_connector = ConnectedAgentTool(
    id=billing_agent.id,
    name="billing_specialist",
    description=(
        "Delegate to this agent for: invoice lookups, payment status checks, "
        "refund requests, billing disputes, and subscription changes."
    ),
)
logistics_connector = ConnectedAgentTool(
    id=logistics_agent.id,
    name="logistics_specialist",
    description=(
        "Delegate to this agent for: shipment tracking, delivery estimates, "
        "lost parcel reports, and address change requests."
    ),
)

orchestrator = client.create_agent(
    model="gpt-4o",
    name="customer-support-orchestrator",
    instructions=(
        "You are a customer support triage agent. Analyse the customer's request and "
        "route it to the appropriate specialist. Do not answer billing or logistics "
        "questions yourself — always delegate to the relevant specialist."
    ),
    tools=[*billing_connector.definitions, *logistics_connector.definitions],
)

# ── 3. Run a conversation ─────────────────────────────────────────────────────
thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="I ordered a laptop last week. Can you track it? Also, invoice INV-8821 seems wrong.",
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=orchestrator.id)
print(f"Run status: {run.status}")

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for content in msg.text_messages:
            print(content.text.value)

# ── 4. Cleanup ─────────────────────────────────────────────────────────────────
client.threads.delete(thread.id)
client.delete_agent(orchestrator.id)
client.delete_agent(billing_agent.id)
client.delete_agent(logistics_agent.id)
```

### Key points

- The `id` parameter must be a real agent ID returned by `create_agent`. The service validates it.
- Each sub-agent is a full agent with its own model, instructions, and tools — they run independently in the service.
- The orchestrator's model decides which sub-agent to call based on the `description` you provide. Write descriptions as routing criteria, not as a prose description of the agent's personality.

---

## 9. `AgentEventHandler` — custom streaming subclass

**Source:** `azure/ai/agents/models/_patch.py` — `AgentEventHandler(BaseAgentEventHandler)`

```python
class AgentEventHandler(BaseAgentEventHandler[...]):
    def __init__(self) -> None: ...
    def set_max_retry(self, max_retry: int) -> None: ...

    # Override any of these in a subclass:
    def on_message_delta(self, delta: MessageDeltaChunk) -> None: ...
    def on_thread_message(self, message: ThreadMessage) -> None: ...
    def on_thread_run(self, run: ThreadRun) -> None: ...
    def on_run_step(self, step: RunStep) -> None: ...
    def on_run_step_delta(self, delta: RunStepDeltaChunk) -> None: ...
    def on_error(self, data: str) -> None: ...
    def on_done(self) -> None: ...
    def on_unhandled_event(self, event_type: str, event_data: Any) -> None: ...
```

Subclass `AgentEventHandler` to react to streaming events as they arrive — useful for progressive rendering, token-level logging, or live UI updates.

### Custom handler with all callbacks

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    AgentEventHandler, MessageDeltaChunk, ThreadMessage,
    ThreadRun, RunStep, RunStepDeltaChunk,
    FunctionTool, MessageRole, RunStatus,
)
from azure.identity import DefaultAzureCredential


class VerboseEventHandler(AgentEventHandler):
    """Logs every streaming event to the console with timestamps."""

    def __init__(self):
        super().__init__()
        self._token_count = 0

    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        """Called for each streamed text token from the assistant."""
        for content_block in delta.delta.content or []:
            if hasattr(content_block, "text") and content_block.text:
                print(content_block.text.value, end="", flush=True)
                self._token_count += 1

    def on_thread_message(self, message: ThreadMessage) -> None:
        """Called when a complete message object arrives (start or completion)."""
        print(f"\n[message] id={message.id} role={message.role} status={message.status}")

    def on_thread_run(self, run: ThreadRun) -> None:
        """Called each time the run's status changes."""
        print(f"[run] status={run.status}")
        if run.status == RunStatus.FAILED:
            print(f"[run] error: {run.last_error}")

    def on_run_step(self, step: RunStep) -> None:
        """Called when a run step is created or completed."""
        print(f"[step] type={step.type} status={step.status}")

    def on_run_step_delta(self, delta: RunStepDeltaChunk) -> None:
        """Called for incremental updates within a run step."""
        # Useful for tracking code interpreter progress
        pass

    def on_error(self, data: str) -> None:
        """Called if the stream emits an error event."""
        print(f"[error] {data}")

    def on_done(self) -> None:
        """Called when the stream signals completion."""
        print(f"\n[done] streamed {self._token_count} token blocks")

    def on_unhandled_event(self, event_type: str, event_data) -> None:
        """Called for any event type not handled by the named callbacks above."""
        print(f"[unhandled] event_type={event_type}")


# ── Usage ─────────────────────────────────────────────────────────────────────
endpoint = os.environ["AZURE_AI_AGENTS_ENDPOINT"]
client = AgentsClient(endpoint=endpoint, credential=DefaultAzureCredential())

def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker symbol."""
    return f"{ticker}: $182.50"

tool = FunctionTool({get_stock_price})
client.enable_auto_function_calls(tools=tool)

agent = client.create_agent(
    model="gpt-4o",
    name="streaming-agent",
    instructions="You are a financial assistant.",
    tools=tool.definitions,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="What's the current price of MSFT and AAPL?",
)

# ── Use the custom handler with runs.stream ───────────────────────────────────
with client.runs.stream(
    thread_id=thread.id,
    agent_id=agent.id,
    event_handler=VerboseEventHandler(),
) as stream:
    stream.until_done()

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

### Lightweight handler for token collection

When you only need the streamed text (e.g. to forward it to a WebSocket):

```python
class TokenCollector(AgentEventHandler):
    def __init__(self):
        super().__init__()
        self.tokens: list[str] = []

    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        for block in delta.delta.content or []:
            if hasattr(block, "text") and block.text:
                self.tokens.append(block.text.value)

    @property
    def text(self) -> str:
        return "".join(self.tokens)


collector = TokenCollector()
with client.runs.stream(
    thread_id=thread.id,
    agent_id=agent.id,
    event_handler=collector,
) as stream:
    stream.until_done()

print(collector.text)
```

---

## 10. `AsyncToolSet` + `AsyncFunctionTool` — concurrent async tool execution

**Source:** `azure/ai/agents/models/_patch.py`

```python
class AsyncFunctionTool(BaseFunctionTool):
    async def execute(self, tool_call: RequiredFunctionToolCall) -> Any:
        function, parsed_arguments = self._get_func_and_args(tool_call)
        if inspect.iscoroutinefunction(function):
            return await function(**parsed_arguments) if parsed_arguments else await function()
        return function(**parsed_arguments) if parsed_arguments else function()

class AsyncToolSet(BaseToolSet):
    def validate_tool_type(self, tool: Tool) -> None:
        if isinstance(tool, FunctionTool):
            raise ValueError(
                "FunctionTool is not supported in AsyncToolSet. "
                "Please use AsyncFunctionTool instead."
            )

    async def execute_tool_calls(self, tool_calls: List[Any]) -> Any:
        tool_outputs = await asyncio.gather(
            *[self._execute_single_tool_call(tc) for tc in tool_calls if tc.type == "function"]
        )
        return tool_outputs
```

`AsyncToolSet` executes all tool calls from a single `requires_action` event **concurrently** using `asyncio.gather`. This is a meaningful performance improvement when the agent issues multiple function calls in parallel and each has I/O latency (database queries, HTTP requests).

### Full async example with concurrent tool calls

```python
import asyncio
import os
from azure.ai.agents.aio import AgentsClient          # async client
from azure.ai.agents.models import (
    AsyncFunctionTool, AsyncToolSet, MessageRole,
)
from azure.identity.aio import DefaultAzureCredential


async def fetch_weather(city: str) -> str:
    """Fetch current weather (async HTTP call)."""
    await asyncio.sleep(0.1)    # simulate I/O
    return f"{city}: 18°C, partly cloudy"


async def fetch_news(topic: str, count: int = 3) -> str:
    """Fetch recent news headlines (async HTTP call)."""
    await asyncio.sleep(0.15)   # simulate I/O
    return f"Top {count} stories about {topic}: [headline1, headline2, headline3]"


def calculate_distance(from_city: str, to_city: str) -> str:
    """Calculate approximate distance between two cities (sync computation)."""
    # AsyncFunctionTool handles sync functions correctly — no await needed
    return f"Distance from {from_city} to {to_city}: approximately 1,200 km"


async def main() -> None:
    endpoint = os.environ["AZURE_AI_AGENTS_ENDPOINT"]

    async with AgentsClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
    ) as client:
        # ── Register mix of async and sync functions ──────────────────────────
        # AsyncFunctionTool auto-detects coroutines via inspect.iscoroutinefunction
        async_func_tool = AsyncFunctionTool({fetch_weather, fetch_news, calculate_distance})

        toolset = AsyncToolSet()
        toolset.add(async_func_tool)

        # ── Enable auto-dispatch on the async client ──────────────────────────
        client.enable_auto_function_calls(tools=toolset)

        agent = await client.create_agent(
            model="gpt-4o",
            name="async-multi-tool-agent",
            instructions="Answer questions using available tools. Call multiple tools in parallel when useful.",
            tools=async_func_tool.definitions,
        )

        thread = await client.threads.create()
        await client.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content=(
                "Fetch the weather for London and Berlin, get 5 news stories about AI, "
                "and tell me the distance from London to Berlin."
            ),
        )

        # create_and_process on the async client handles tool dispatch concurrently
        run = await client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
        print(f"Status: {run.status}")

        # Read response
        async for msg in client.messages.list(thread_id=thread.id):
            if msg.role == "assistant":
                for content in msg.text_messages:
                    print(content.text.value)

        # Cleanup
        await client.threads.delete(thread.id)
        await client.delete_agent(agent.id)


asyncio.run(main())
```

### `AsyncToolSet` vs `ToolSet` — when to use which

| Scenario | Use |
|---|---|
| Your agent code is sync (`def`, not `async def`) | `ToolSet` + `FunctionTool` |
| Your tool functions use `await` (aiohttp, asyncpg, etc.) | `AsyncToolSet` + `AsyncFunctionTool` |
| You want concurrent tool execution | `AsyncToolSet` — `asyncio.gather` runs all parallel tool calls at once |
| You have a mix of sync and async tools | `AsyncToolSet` + `AsyncFunctionTool` — handles both correctly |
| You accidentally add `FunctionTool` to `AsyncToolSet` | `ValueError` is raised immediately at `toolset.add()` time |

### Why concurrent execution matters

When the model issues `N` parallel function calls in one `requires_action` event:

- **`ToolSet`** runs them sequentially — total wall time is `sum(latency of each call)`.
- **`AsyncToolSet`** runs them with `asyncio.gather` — total wall time is `max(latency of each call)`.

For 3 tool calls each taking 200 ms, sequential execution is 600 ms; concurrent is ~200 ms.

---

## Patterns Combining Multiple Classes

### Pattern A — Grounded RAG agent (FileSearch + AzureAISearch + BingGrounding)

```python
from azure.ai.agents.models import ToolSet, FileSearchTool, AzureAISearchTool, AzureAISearchQueryType, BingGroundingTool

toolset = ToolSet()
toolset.add(FileSearchTool(["vs_internal_docs"]))        # proprietary internal docs
toolset.add(AzureAISearchTool(                           # structured product catalogue
    index_connection_id=SEARCH_CONNECTION_ID,
    index_name="products",
    query_type=AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
    top_k=5,
))
toolset.add(BingGroundingTool(                           # live web context
    connection_id=BING_CONNECTION_ID,
    market="en-GB",
    freshness="Week",
    count=5,
))

agent = client.create_agent(
    model="gpt-4o",
    name="grounded-rag-agent",
    instructions=(
        "Answer questions using internal documents first, then the product catalogue, "
        "and supplement with live web results only if the internal sources are insufficient."
    ),
    toolset=toolset,
)
```

### Pattern B — Multi-agent with streaming (ConnectedAgentTool + AgentEventHandler)

```python
class StreamingOrchestratorHandler(AgentEventHandler):
    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        for block in delta.delta.content or []:
            if hasattr(block, "text") and block.text:
                # Forward tokens to your WebSocket or SSE endpoint
                print(block.text.value, end="", flush=True)

handler = StreamingOrchestratorHandler()
with client.runs.stream(
    thread_id=thread.id,
    agent_id=orchestrator.id,
    event_handler=handler,
) as stream:
    stream.until_done()
```

### Pattern C — Async multi-agent (AsyncToolSet + create_thread_and_process_run)

```python
async def run_all_regions(client: AgentsClient, question: str) -> dict[str, str]:
    """Ask the same question to region-specific agents concurrently."""
    tasks = [
        client.create_thread_and_process_run(
            agent_id=ag.id,
            thread=AgentThreadCreationOptions(
                messages=[ThreadMessageOptions(role=MessageRole.USER, content=question)]
            ),
        )
        for ag in regional_agents
    ]
    runs = await asyncio.gather(*tasks)
    return {run.thread_id: run.status for run in runs}
```

---

## Revision history

| Date | Change |
|------|--------|
| 2026-05-28 | Initial release — 10 source-verified classes. All signatures and behaviour verified against `azure-ai-agents==1.1.0` installed at `/usr/local/lib/python3.11/dist-packages/azure/ai/agents/`. |

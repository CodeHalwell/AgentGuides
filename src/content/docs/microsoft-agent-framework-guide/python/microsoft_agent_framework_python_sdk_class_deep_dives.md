---
title: "Microsoft Azure AI Agents SDK (Python) — Class Deep Dives v1.1.0"
description: "Source-verified deep dives into 10 key classes from azure-ai-agents 1.1.0: AgentsClient, FunctionTool, ToolSet, CodeInterpreterTool, FileSearchTool, BingGroundingTool, ConnectedAgentTool, AgentEventHandler, ThreadMessage, and OpenApiTool."
framework: microsoft-agent-framework
language: python
---

# Microsoft Azure AI Agents SDK (Python) — Class Deep Dives v1.1.0

This document provides source-verified, in-depth reference material for the **`azure-ai-agents`** Python SDK at version 1.1.0. Every class signature, method, and property described here has been validated directly against the library source code. If you have seen documentation referring to a package called `agent-framework`, that package is incorrect — the real package name is `azure-ai-agents`.

> **Package**: `azure-ai-agents`  
> **Version covered**: 1.1.0  
> **Main import**: `from azure.ai.agents import AgentsClient`  
> **Async import**: `from azure.ai.agents.aio import AgentsClient`

---

## Table of Contents

1. [Installation](#installation)
2. [End-to-End Quickstart](#end-to-end-quickstart)
3. [AgentsClient](#agentsclient)
4. [ThreadMessage](#threadmessage)
5. [FunctionTool](#functiontool)
6. [ToolSet](#toolset)
7. [CodeInterpreterTool](#codeinterpretertool)
8. [FileSearchTool](#filesearchtool)
9. [BingGroundingTool](#bingroundingtool)
10. [ConnectedAgentTool](#connectedagenttool)
11. [AgentEventHandler (Streaming)](#agenteventhandler-streaming)
12. [OpenApiTool](#openapitool)
13. [Async Variants Summary](#async-variants-summary)
14. [Common Patterns and Gotchas](#common-patterns-and-gotchas)

---

## Installation

```bash
pip install azure-ai-agents
```

The SDK depends on `azure-identity` for authentication, so install it too if you do not already have it:

```bash
pip install azure-ai-agents azure-identity
```

**Do not install** `agent-framework` — that is a different, unrelated package and will not provide the classes described in this document.

---

## End-to-End Quickstart

This section demonstrates the complete lifecycle of an agent interaction: installation, client creation, agent creation, thread and message creation, running the agent, polling for completion, reading the response, and cleaning up.

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import MessageRole
from azure.identity import DefaultAzureCredential

# ── 1. Construct the client ──────────────────────────────────────────────────
# The endpoint follows the pattern:
#   https://<aiservices-id>.services.ai.azure.com/api/projects/<project-name>
endpoint = os.environ["AZURE_AI_AGENTS_ENDPOINT"]

client = AgentsClient(
    endpoint=endpoint,
    credential=DefaultAzureCredential(),
)

# ── 2. Create an agent ───────────────────────────────────────────────────────
agent = client.create_agent(
    model="gpt-4o",
    name="quickstart-agent",
    instructions=(
        "You are a helpful assistant. Answer questions clearly and concisely."
    ),
)
print(f"Created agent: {agent.id}")

# ── 3. Create a conversation thread ─────────────────────────────────────────
thread = client.threads.create()
print(f"Created thread: {thread.id}")

# ── 4. Add a user message to the thread ─────────────────────────────────────
message = client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="What is the capital of Portugal, and what is it famous for?",
)
print(f"Created message: {message.id}")

# ── 5. Run the agent and poll until completion ───────────────────────────────
# create_and_process polls internally — it blocks until the run finishes.
run = client.runs.create_and_process(
    thread_id=thread.id,
    agent_id=agent.id,
)
print(f"Run finished with status: {run.status}")

# ── 6. Read the assistant's response ────────────────────────────────────────
messages = client.messages.list(thread_id=thread.id)
for msg in messages:
    if msg.role == "assistant":
        for text_content in msg.text_messages:
            print(f"Assistant: {text_content.text.value}")

# ── 7. Cleanup ───────────────────────────────────────────────────────────────
client.threads.delete(thread.id)
client.delete_agent(agent.id)
print("Cleaned up thread and agent.")
```

### Async version of the same quickstart

```python
import asyncio
import os
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import MessageRole
from azure.identity.aio import DefaultAzureCredential


async def main() -> None:
    endpoint = os.environ["AZURE_AI_AGENTS_ENDPOINT"]

    async with AgentsClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
    ) as client:
        # Create agent
        agent = await client.create_agent(
            model="gpt-4o",
            name="async-quickstart-agent",
            instructions="You are a helpful assistant.",
        )

        # Create thread and message
        thread = await client.threads.create()
        await client.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content="Summarise the key events of the Apollo 11 mission.",
        )

        # Run and poll
        run = await client.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent.id,
        )
        print(f"Status: {run.status}")

        # Read response
        messages = await client.messages.list(thread_id=thread.id)
        async for msg in messages:
            if msg.role == "assistant":
                for text_content in msg.text_messages:
                    print(text_content.text.value)

        # Cleanup
        await client.threads.delete(thread.id)
        await client.delete_agent(agent.id)


asyncio.run(main())
```

---

## AgentsClient

`AgentsClient` is the central entry point for all operations in the `azure-ai-agents` SDK. It manages agents, threads, messages, runs, files, and vector stores through a combination of top-level methods and sub-operation namespaces.

### Import

```python
# Synchronous
from azure.ai.agents import AgentsClient

# Asynchronous
from azure.ai.agents.aio import AgentsClient
```

### Constructor

```python
AgentsClient(
    endpoint: str,
    credential: TokenCredential,
    **kwargs,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `endpoint` | `str` | Full project endpoint URL. Format: `https://<aiservices-id>.services.ai.azure.com/api/projects/<project-name>` |
| `credential` | `TokenCredential` | Any `azure-identity` credential object, e.g. `DefaultAzureCredential()` |
| `**kwargs` | | Optional keyword arguments passed to the underlying HTTP transport (e.g. `api_version`, `retry_policy`) |

The endpoint URL must include both the AI services resource ID and the project name. Omitting either component will result in authentication or routing errors.

### Agent management methods

#### `create_agent`

```python
client.create_agent(
    *,
    model: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    tools: Optional[List[ToolDefinition]] = None,
    tool_resources: Optional[ToolResources] = None,
    toolset: Optional[ToolSet] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    response_format: Optional[AgentsResponseFormat] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Agent
```

Creates a persistent agent in the Azure AI Agents service. The agent object is stored server-side and can be reused across many threads and sessions.

- **`model`** is the only required argument — it specifies which deployed model backs the agent.
- **`toolset`** is a convenience shortcut: if provided, the SDK automatically extracts `tools` and `tool_resources` from the `ToolSet` object. Do not pass both `toolset` and `tools`/`tool_resources` simultaneously.
- **`instructions`** serve as the system prompt and define the agent's persona, capabilities, and constraints.
- **`metadata`** is a dict of up to 16 key-value pairs (both key and value must be strings) for your own tracking purposes.

#### `get_agent`

```python
client.get_agent(agent_id: str) -> Agent
```

Retrieves an existing agent by its ID. Useful when you want to attach tools to an agent that was previously created or to inspect its configuration.

#### `list_agents`

```python
client.list_agents() -> Iterable[Agent]
```

Returns an iterable of all agents in the current project. Supports standard iteration.

#### `update_agent`

```python
client.update_agent(
    agent_id: str,
    *,
    model: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    tools: Optional[List[ToolDefinition]] = None,
    tool_resources: Optional[ToolResources] = None,
    toolset: Optional[ToolSet] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    response_format: Optional[AgentsResponseFormat] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Agent
```

Updates one or more properties of an existing agent. Only the fields you supply are changed; omitted fields retain their current values.

#### `delete_agent`

```python
client.delete_agent(agent_id: str) -> AgentDeletionStatus
```

Permanently deletes an agent from the service.

#### `enable_auto_function_calls`

```python
client.enable_auto_function_calls(functions: Set[Callable]) -> None
```

Registers a set of Python callables on the client so that when a run enters `REQUIRES_ACTION` state for function tool calls, the SDK automatically executes the matching Python function and submits the results — without you having to implement the polling loop. Use this method when you prefer simplicity over fine-grained control.

### Convenience run methods

#### `create_thread_and_process_run`

```python
client.create_thread_and_process_run(
    *,
    agent_id: str,
    thread: Optional[AgentThreadCreationOptions] = None,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
    toolset: Optional[ToolSet] = None,
    ...
) -> ThreadRun
```

Creates a new thread and immediately runs the agent on it, polling until the run reaches a terminal state. This is the highest-level convenience method — suitable for single-turn interactions where you do not need to inspect intermediate states.

#### `create_thread_and_run`

```python
client.create_thread_and_run(...) -> ThreadRun
```

Same as above but does **not** poll for completion. Returns the `ThreadRun` immediately after the run is initiated. You must poll `client.runs.get()` yourself.

### Sub-operation namespaces

| Attribute | Purpose |
|-----------|---------|
| `client.threads` | Create, get, list, update, delete threads |
| `client.messages` | Create, list, get messages within threads |
| `client.runs` | Create, poll, stream, cancel, submit tool outputs for runs |
| `client.run_steps` | Inspect individual steps within a run |
| `client.files` | Upload and manage files for use with tools |
| `client.vector_stores` | Create and manage vector stores for file search |
| `client.vector_store_files` | Attach files to vector stores |
| `client.vector_store_file_batches` | Batch-add files to vector stores |

### Example: Full agent lifecycle with update

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import MessageRole
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# Create with minimal options
agent = client.create_agent(
    model="gpt-4o",
    name="finance-analyst",
    instructions="You analyse financial data and provide insights.",
    temperature=0.3,
)
print(f"Agent created: {agent.id}, model: {agent.model}")

# Inspect existing agents
for existing in client.list_agents():
    print(f"  - {existing.id}: {existing.name}")

# Update instructions at runtime
updated = client.update_agent(
    agent.id,
    instructions=(
        "You analyse financial data and provide insights. "
        "Always cite the source data in your response."
    ),
    metadata={"team": "finance", "env": "production"},
)
print(f"Instructions updated: {updated.instructions[:60]}...")

# Retrieve agent by ID (simulating a separate session)
retrieved = client.get_agent(agent.id)
print(f"Retrieved agent name: {retrieved.name}")

# Cleanup
client.delete_agent(agent.id)
```

### Example: Threads and messages sub-operations

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import MessageRole
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

agent = client.create_agent(model="gpt-4o", name="demo-agent",
                             instructions="Be helpful and concise.")

# Threads sub-operation
thread = client.threads.create()

# Prepopulate the thread with context
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="Context: we are discussing renewable energy policy in the UK.",
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
print(f"Run status: {run.status}")

# List all messages in the thread (most recent first by default)
all_messages = client.messages.list(thread_id=thread.id)
for m in all_messages:
    role_label = "User" if m.role == "user" else "Assistant"
    for tc in m.text_messages:
        print(f"[{role_label}] {tc.text.value}")

# Retrieve a specific thread later
same_thread = client.threads.get(thread.id)
print(f"Thread retrieved: {same_thread.id}")

# Cleanup
client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

### Example: auto function calls

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import FunctionTool, ToolSet, MessageRole
from azure.identity import DefaultAzureCredential


def get_weather(city: str, unit: str = "celsius") -> str:
    """
    Return current weather for a city.

    :param city: Name of the city.
    :param unit: Temperature unit — 'celsius' or 'fahrenheit'.
    """
    # Real implementation would call a weather API
    return f"The weather in {city} is 18°{unit[0].upper()} and partly cloudy."


client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# Register functions for automatic execution
client.enable_auto_function_calls(functions={get_weather})

toolset = ToolSet()
toolset.add(FunctionTool(functions={get_weather}))

agent = client.create_agent(
    model="gpt-4o",
    name="weather-agent",
    instructions="You help users check weather. Use the get_weather tool.",
    toolset=toolset,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="What is the weather like in Edinburgh right now?",
)

# enable_auto_function_calls means tool calls are handled transparently
run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

messages = client.messages.list(thread_id=thread.id)
for msg in messages:
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(tc.text.value)

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

---

## ThreadMessage

`ThreadMessage` represents a single message in a conversation thread. It may come from a user or from the assistant.

### Import

```python
from azure.ai.agents.models import ThreadMessage, MessageRole
```

### Key properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique message identifier |
| `thread_id` | `str` | ID of the containing thread |
| `role` | `str` | `"user"` or `"assistant"` |
| `content` | `List[MessageContent]` | Raw content list (heterogeneous types) |
| `status` | `MessageStatus` | Lifecycle status of the message |
| `created_at` | `datetime` | Creation timestamp |

### Convenience properties (added in `_patch.py`)

These properties were added by Microsoft engineers in `_patch.py` to make message content extraction far more ergonomic than iterating the raw `content` list:

| Property | Return type | Description |
|----------|-------------|-------------|
| `text_messages` | `List[MessageTextContent]` | All text content items |
| `image_contents` | `List[MessageImageFileContent]` | All image file content items |
| `file_citation_annotations` | `List[MessageTextFileCitationAnnotation]` | Inline file citation annotations |
| `file_path_annotations` | `List[MessageTextFilePathAnnotation]` | File path annotations |
| `url_citation_annotations` | `List[MessageTextUrlCitationAnnotation]` | URL citation annotations |

### Accessing text content

```python
# Preferred approach — use convenience properties
for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for text_content in msg.text_messages:
            # text_content is MessageTextContent
            print(text_content.text.value)          # The actual text string
            print(text_content.text.annotations)    # List of annotation objects
```

### Example: Extracting all content types

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import MessageRole, FileSearchTool
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# Assume an agent with file search is already set up
agent = client.create_agent(
    model="gpt-4o",
    name="content-demo",
    instructions="Answer questions from the uploaded documents.",
)
thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="Summarise the main findings.",
)
run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

messages = client.messages.list(thread_id=thread.id)
for msg in messages:
    print(f"\n--- Message {msg.id} [{msg.role}] ---")

    # Text content
    for tc in msg.text_messages:
        print(f"Text: {tc.text.value}")

    # File citations (present when file search tool is used)
    for citation in msg.file_citation_annotations:
        print(f"Citation: file_id={citation.file_citation.file_id}, "
              f"quote='{citation.file_citation.quote}'")

    # URL citations (present when Bing Grounding is used)
    for url_ann in msg.url_citation_annotations:
        print(f"URL citation: {url_ann.url_citation.url} — {url_ann.url_citation.title}")

    # Image outputs (present when code interpreter generates plots)
    for img in msg.image_contents:
        print(f"Image file ID: {img.image_file.file_id}")

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

### Example: Filtering and ordering messages

```python
# List messages with ordering — list() returns newest-first by default
# To get oldest-first, specify order parameter
from azure.ai.agents.models import ListSortOrder

messages = client.messages.list(
    thread_id=thread.id,
    order=ListSortOrder.ASCENDING,
    limit=20,
)

# Get a specific message by ID
specific_msg = client.messages.get(
    thread_id=thread.id,
    message_id=message.id,
)
print(f"Specific message role: {specific_msg.role}")
print(f"Specific message status: {specific_msg.status}")
```

---

## FunctionTool

`FunctionTool` enables agents to call your Python functions. It introspects function signatures and docstrings automatically to build the JSON schema sent to the model.

### Import

```python
from azure.ai.agents.models import FunctionTool          # sync
from azure.ai.agents.models import AsyncFunctionTool     # async
```

### Class signature

```python
class FunctionTool(BaseFunctionTool):
    def __init__(self, functions: Set[Callable[..., Any]])
```

### Key methods and properties

| Member | Signature | Description |
|--------|-----------|-------------|
| `add_functions` | `(extra_functions: Set[Callable])` | Adds more functions after construction |
| `execute` | `(tool_call: RequiredFunctionToolCall) -> Any` | Called automatically by the SDK during run processing |
| `definitions` | property → `List[FunctionToolDefinition]` | JSON schema definitions sent to the model |
| `resources` | property → `ToolResources` | Tool resource descriptor (empty for function tools) |

### Docstring format for parameter descriptions

The SDK parses docstrings to extract parameter descriptions. Use the `:param name: description` format (Sphinx/reStructuredText style):

```python
def calculate_mortgage(
    principal: float,
    annual_rate: float,
    years: int,
) -> str:
    """
    Calculate monthly mortgage repayment.

    :param principal: Loan amount in GBP.
    :param annual_rate: Annual interest rate as a percentage (e.g. 3.5 for 3.5%).
    :param years: Loan duration in years.
    """
    monthly_rate = (annual_rate / 100) / 12
    n = years * 12
    if monthly_rate == 0:
        payment = principal / n
    else:
        payment = principal * (monthly_rate * (1 + monthly_rate) ** n) / ((1 + monthly_rate) ** n - 1)
    return f"Monthly payment: £{payment:.2f}"
```

### Example: Basic function tool usage

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import FunctionTool, ToolSet, MessageRole
from azure.identity import DefaultAzureCredential


def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """
    Get the current exchange rate between two currencies.

    :param from_currency: The source currency code (e.g. 'GBP').
    :param to_currency: The target currency code (e.g. 'EUR').
    """
    # Simulated — replace with real FX API call
    rates = {"GBP_EUR": 1.17, "GBP_USD": 1.27, "EUR_USD": 1.08}
    key = f"{from_currency}_{to_currency}"
    rate = rates.get(key, "unknown")
    return f"1 {from_currency} = {rate} {to_currency}"


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert an amount from one currency to another.

    :param amount: The amount to convert.
    :param from_currency: Source currency code.
    :param to_currency: Target currency code.
    """
    rates = {"GBP_EUR": 1.17, "GBP_USD": 1.27}
    key = f"{from_currency}_{to_currency}"
    rate = rates.get(key, 1.0)
    converted = amount * rate
    return f"{amount} {from_currency} = {converted:.2f} {to_currency}"


client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# Create the tool with both functions
fx_tool = FunctionTool(functions={get_exchange_rate, convert_currency})

# Inspect what definitions look like (useful for debugging)
for defn in fx_tool.definitions:
    print(f"Function: {defn.function.name}")
    print(f"  Schema: {defn.function.parameters}")

toolset = ToolSet()
toolset.add(fx_tool)

agent = client.create_agent(
    model="gpt-4o",
    name="fx-agent",
    instructions="Help users with currency conversions and exchange rates.",
    toolset=toolset,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="How much is £500 in euros?",
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(tc.text.value)

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

### Example: Adding functions after construction

```python
# Start with one function
tool = FunctionTool(functions={get_exchange_rate})

# Add more functions later (useful when building tools dynamically)
tool.add_functions({convert_currency})

# Now both functions are registered
print(f"Registered functions: {len(tool.definitions)}")
```

### Example: Async function tool

```python
import asyncio
import os
import httpx
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import AsyncFunctionTool, AsyncToolSet, MessageRole
from azure.identity.aio import DefaultAzureCredential


async def fetch_stock_price(ticker: str) -> str:
    """
    Fetch the current stock price for a given ticker symbol.

    :param ticker: Stock ticker symbol (e.g. 'MSFT', 'AAPL').
    """
    # In production, call a real stock price API here
    mock_prices = {"MSFT": 415.23, "AAPL": 189.45, "GOOGL": 178.90}
    price = mock_prices.get(ticker.upper(), "N/A")
    return f"{ticker.upper()}: ${price}"


async def main() -> None:
    async with AgentsClient(
        endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
        credential=DefaultAzureCredential(),
    ) as client:
        stock_tool = AsyncFunctionTool(functions={fetch_stock_price})

        toolset = AsyncToolSet()
        toolset.add(stock_tool)

        agent = await client.create_agent(
            model="gpt-4o",
            name="stock-agent",
            instructions="Help users look up stock prices.",
            toolset=toolset,
        )

        thread = await client.threads.create()
        await client.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content="What is the current price of Microsoft stock?",
        )

        run = await client.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent.id,
        )

        messages = await client.messages.list(thread_id=thread.id)
        async for msg in messages:
            if msg.role == "assistant":
                for tc in msg.text_messages:
                    print(tc.text.value)

        await client.threads.delete(thread.id)
        await client.delete_agent(agent.id)


asyncio.run(main())
```

### Gotchas

- Functions must have type annotations. The SDK uses these to generate the JSON schema's `type` fields.
- Return values must be JSON-serialisable strings, dicts, or primitives. Returning complex objects may cause serialisation errors.
- The `:param name: description` docstring format is critical — without it, the model receives no parameter descriptions and may misuse the function.
- Do not mix `AsyncFunctionTool` into a synchronous `ToolSet`; it will raise an error at runtime.

---

## ToolSet

`ToolSet` is a container that aggregates multiple tool types and passes them to an agent or run in a single object. It enforces the constraint that each tool type may appear only once.

### Import

```python
from azure.ai.agents.models import ToolSet       # sync
from azure.ai.agents.models import AsyncToolSet  # async
```

### Class signature

```python
class ToolSet:
    def add(self, tool: Tool) -> None
    def remove(self, tool_type: Type[Tool]) -> None
    def get_tool(self, tool_type: Type[Tool]) -> Tool
    def execute_tool_calls(self, tool_calls: List[Any]) -> Any
    
    @property
    def definitions(self) -> List[ToolDefinition]
    
    @property
    def resources(self) -> ToolResources
```

### Rules

- You cannot add two tools of the same type. Adding a second `FunctionTool` replaces the first.
- `AsyncFunctionTool` is **not** allowed in a synchronous `ToolSet`. Use `AsyncToolSet` for async scenarios.
- `definitions` and `resources` are computed dynamically from all registered tools.

### Example: Multi-tool agent

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    FunctionTool,
    CodeInterpreterTool,
    FileSearchTool,
    ToolSet,
    MessageRole,
)
from azure.identity import DefaultAzureCredential


def lookup_employee(employee_id: str) -> str:
    """
    Look up an employee record by ID.

    :param employee_id: The employee's unique identifier.
    """
    records = {
        "E001": {"name": "Alice Smith", "department": "Engineering", "level": "Senior"},
        "E002": {"name": "Bob Jones", "department": "Finance", "level": "Manager"},
    }
    record = records.get(employee_id)
    if record:
        return str(record)
    return f"Employee {employee_id} not found."


client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# Upload a file for code interpreter
with open("data.csv", "rb") as f:
    uploaded_file = client.files.upload(file=f, purpose="assistants")

# Build a toolset with three tools
toolset = ToolSet()
toolset.add(FunctionTool(functions={lookup_employee}))
toolset.add(CodeInterpreterTool(file_ids=[uploaded_file.id]))
toolset.add(FileSearchTool())

# Inspect combined definitions
print(f"Total tool definitions: {len(toolset.definitions)}")
for d in toolset.definitions:
    print(f"  - {type(d).__name__}")

agent = client.create_agent(
    model="gpt-4o",
    name="multi-tool-agent",
    instructions=(
        "You can look up employees, run code on uploaded CSV data, "
        "and search documents."
    ),
    toolset=toolset,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="Look up employee E001 and tell me their department.",
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(tc.text.value)

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

### Example: Dynamically modifying a toolset

```python
from azure.ai.agents.models import (
    ToolSet, FunctionTool, CodeInterpreterTool, BingGroundingTool
)

toolset = ToolSet()
toolset.add(CodeInterpreterTool())

# Add Bing grounding
bing = BingGroundingTool(connection_id=os.environ["BING_CONNECTION_ID"])
toolset.add(bing)

# Later, remove Bing and add a function tool instead
toolset.remove(BingGroundingTool)
toolset.add(FunctionTool(functions={my_function}))

# Retrieve a specific tool to inspect or modify it
code_tool = toolset.get_tool(CodeInterpreterTool)
code_tool.add_file("file-abc123")
```

---

## CodeInterpreterTool

`CodeInterpreterTool` enables the agent to write and execute Python code in a sandboxed environment. It can produce charts, parse data, and generate downloadable files.

### Import

```python
from azure.ai.agents.models import CodeInterpreterTool
```

### Class signature

```python
class CodeInterpreterTool:
    def __init__(self, file_ids: Optional[List[str]] = None)
    
    def add_file(self, file_id: str) -> None
    def remove_file(self, file_id: str) -> None
    
    @property
    def definitions(self) -> List[CodeInterpreterToolDefinition]
    
    @property
    def resources(self) -> ToolResources  # Returns CodeInterpreterToolResource
```

### Example: Data analysis with code interpreter

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import CodeInterpreterTool, ToolSet, MessageRole
from azure.identity import DefaultAzureCredential
import io

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# Create a sample CSV in memory and upload it
csv_content = b"""month,revenue,costs
Jan,120000,85000
Feb,135000,90000
Mar,148000,92000
Apr,162000,95000
May,175000,98000
"""
csv_file = io.BytesIO(csv_content)
csv_file.name = "revenue.csv"

uploaded = client.files.upload(file=csv_file, purpose="assistants")
print(f"Uploaded file: {uploaded.id}")

# Create tool with the file attached
code_tool = CodeInterpreterTool(file_ids=[uploaded.id])
toolset = ToolSet()
toolset.add(code_tool)

agent = client.create_agent(
    model="gpt-4o",
    name="data-analyst",
    instructions=(
        "You are a data analyst. Use the code interpreter to analyse data, "
        "produce statistics, and generate charts when helpful."
    ),
    toolset=toolset,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content=(
        "Analyse the revenue.csv file. Calculate total revenue, total costs, "
        "and net profit for each month. Then identify the month with the highest "
        "profit margin."
    ),
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
print(f"Run status: {run.status}")

messages = client.messages.list(thread_id=thread.id)
for msg in messages:
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(tc.text.value)
        # Download any generated image files
        for img in msg.image_contents:
            print(f"Generated image file ID: {img.image_file.file_id}")
            # To download: client.files.get_content(img.image_file.file_id)

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

### Example: Adding files to code interpreter after construction

```python
# Start with no files
code_tool = CodeInterpreterTool()

# Upload and attach files later
file_a = client.files.upload(file=open("dataset_a.csv", "rb"), purpose="assistants")
file_b = client.files.upload(file=open("dataset_b.csv", "rb"), purpose="assistants")

code_tool.add_file(file_a.id)
code_tool.add_file(file_b.id)

# Remove a file if no longer needed
code_tool.remove_file(file_a.id)

toolset = ToolSet()
toolset.add(code_tool)
```

### Gotchas

- Files must be uploaded with `purpose="assistants"` before being attached to `CodeInterpreterTool`.
- The tool's sandbox runs Python only; it cannot execute arbitrary shell commands.
- Generated files (charts, processed CSVs) are available via their file IDs in `image_contents` of the response message. Use `client.files.get_content(file_id)` to download them.

---

## FileSearchTool

`FileSearchTool` enables agents to perform semantic search over documents you have uploaded and indexed in a vector store. It is ideal for retrieval-augmented generation (RAG) scenarios.

### Import

```python
from azure.ai.agents.models import FileSearchTool
```

### Class signature

```python
class FileSearchTool:
    def __init__(self, vector_store_ids: Optional[List[str]] = None)
    
    def add_vector_store(self, store_id: str) -> None
    def remove_vector_store(self, store_id: str) -> None
    
    @property
    def definitions(self) -> List[FileSearchToolDefinition]
    
    @property
    def resources(self) -> ToolResources  # Returns FileSearchToolResource
```

### Example: Full RAG setup with vector store

```python
import os
import time
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import FileSearchTool, ToolSet, MessageRole
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# Upload documents
with open("company_policy.pdf", "rb") as f:
    policy_file = client.files.upload(file=f, purpose="assistants")

with open("employee_handbook.pdf", "rb") as f:
    handbook_file = client.files.upload(file=f, purpose="assistants")

print(f"Uploaded files: {policy_file.id}, {handbook_file.id}")

# Create vector store
vector_store = client.vector_stores.create(
    name="company-docs",
    file_ids=[policy_file.id, handbook_file.id],
)

# Wait for the vector store to finish processing
while vector_store.status == "in_progress":
    time.sleep(2)
    vector_store = client.vector_stores.get(vector_store.id)

print(f"Vector store status: {vector_store.status}")

# Create the file search tool
search_tool = FileSearchTool(vector_store_ids=[vector_store.id])
toolset = ToolSet()
toolset.add(search_tool)

agent = client.create_agent(
    model="gpt-4o",
    name="policy-assistant",
    instructions=(
        "You are a company policy expert. Search the uploaded documents "
        "to answer employee questions accurately. Always cite your sources."
    ),
    toolset=toolset,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="What is the company's policy on remote working arrangements?",
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

messages = client.messages.list(thread_id=thread.id)
for msg in messages:
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(tc.text.value)
        # File citations tell you which document and passage was referenced
        for citation in msg.file_citation_annotations:
            print(f"\n  [Citation] File: {citation.file_citation.file_id}")
            if hasattr(citation.file_citation, 'quote'):
                print(f"  Quote: {citation.file_citation.quote}")

# Cleanup
client.threads.delete(thread.id)
client.delete_agent(agent.id)
client.vector_stores.delete(vector_store.id)
client.files.delete(policy_file.id)
client.files.delete(handbook_file.id)
```

### Example: Adding vector stores dynamically

```python
# Create tool with no initial stores
search_tool = FileSearchTool()

# Add stores later
search_tool.add_vector_store("vs_abc123")
search_tool.add_vector_store("vs_def456")

# Remove a store
search_tool.remove_vector_store("vs_abc123")
```

---

## BingGroundingTool

`BingGroundingTool` gives agents real-time access to web search results via Bing. The agent can retrieve up-to-date information that was not present in its training data.

### Import

```python
from azure.ai.agents.models import BingGroundingTool
```

### Class signature

```python
class BingGroundingTool:
    def __init__(
        self,
        connection_id: str,
        market: str = "",
        set_lang: str = "",
        count: int = 5,
        freshness: str = "",
    )
    
    @property
    def definitions(self) -> List[BingGroundingToolDefinition]
    
    @property
    def resources(self) -> ToolResources  # Empty — Bing uses no tool resources
```

| Parameter | Description |
|-----------|-------------|
| `connection_id` | Azure AI Foundry connection ID for the Bing resource (not the API key directly) |
| `market` | Bing market code, e.g. `"en-GB"` for UK English results |
| `set_lang` | UI language for Bing result metadata |
| `count` | Number of results to return (default 5) |
| `freshness` | Filter results by age: `"Day"`, `"Week"`, `"Month"` |

### Example: Web-grounded research agent

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import BingGroundingTool, ToolSet, MessageRole
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

bing_tool = BingGroundingTool(
    connection_id=os.environ["BING_CONNECTION_ID"],
    market="en-GB",
    count=8,
    freshness="Week",   # Only results from the past week
)

toolset = ToolSet()
toolset.add(bing_tool)

agent = client.create_agent(
    model="gpt-4o",
    name="research-agent",
    instructions=(
        "You are a research assistant with access to live web search. "
        "Always cite the URLs of sources you use in your response."
    ),
    toolset=toolset,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content=(
        "What are the latest developments in the UK's AI regulation framework? "
        "Summarise the key points from recent news."
    ),
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

messages = client.messages.list(thread_id=thread.id)
for msg in messages:
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(tc.text.value)
        # URL citations from Bing results
        for url_ann in msg.url_citation_annotations:
            print(f"\n  Source: {url_ann.url_citation.title}")
            print(f"  URL: {url_ann.url_citation.url}")

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

### Gotchas

- The `connection_id` is the Azure AI Foundry connection identifier (e.g. `"/subscriptions/.../connections/bing-search"`), not a raw Bing API key.
- `BingGroundingTool` has no `resources` to attach to the agent — the connection is resolved entirely through the `definitions`.
- URL citations appear on the `ThreadMessage` object via `msg.url_citation_annotations`, not in the raw text.

---

## ConnectedAgentTool

`ConnectedAgentTool` enables multi-agent architectures where one agent (the orchestrator) can delegate tasks to another agent (a sub-agent). The orchestrator uses the description to decide when to invoke the sub-agent.

### Import

```python
from azure.ai.agents.models import ConnectedAgentTool
```

### Class signature

```python
class ConnectedAgentTool:
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
    )
    
    @property
    def definitions(self) -> List[ConnectedAgentToolDefinition]
    
    @property
    def resources(self) -> ToolResources  # Empty
```

| Parameter | Description |
|-----------|-------------|
| `id` | The ID of the sub-agent (obtained from `agent.id` when you create it) |
| `name` | A short identifier for the sub-agent |
| `description` | A clear description of what the sub-agent does; the orchestrator uses this to decide when to delegate |

### Example: Orchestrator with two specialised sub-agents

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    ConnectedAgentTool, FunctionTool, ToolSet, MessageRole
)
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# ── Sub-agent 1: Legal specialist ────────────────────────────────────────────
legal_agent = client.create_agent(
    model="gpt-4o",
    name="legal-specialist",
    instructions=(
        "You are a legal expert specialising in UK contract law. "
        "Provide clear, accurate legal analysis."
    ),
)

# ── Sub-agent 2: Finance specialist ─────────────────────────────────────────
finance_agent = client.create_agent(
    model="gpt-4o",
    name="finance-specialist",
    instructions=(
        "You are a financial analyst. Analyse financial terms, valuations, "
        "and commercial viability."
    ),
)

# ── Orchestrator agent ───────────────────────────────────────────────────────
legal_tool = ConnectedAgentTool(
    id=legal_agent.id,
    name="legal_specialist",
    description=(
        "Handles legal questions about contracts, liabilities, compliance, "
        "and UK law. Use for any legal analysis required."
    ),
)

finance_tool = ConnectedAgentTool(
    id=finance_agent.id,
    name="finance_specialist",
    description=(
        "Handles financial analysis including valuations, pricing terms, "
        "ROI calculations, and commercial risk assessment."
    ),
)

toolset = ToolSet()
toolset.add(legal_tool)
toolset.add(finance_tool)

orchestrator = client.create_agent(
    model="gpt-4o",
    name="deal-orchestrator",
    instructions=(
        "You coordinate contract review requests. Delegate legal questions to "
        "the legal specialist and financial questions to the finance specialist. "
        "Synthesise their answers into a final recommendation."
    ),
    toolset=toolset,
)

# ── Run the orchestration ─────────────────────────────────────────────────────
thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content=(
        "We have a software licensing deal worth £2M over 3 years with a "
        "liability cap of £500k. Is the liability cap standard under UK law, "
        "and is £2M reasonable for enterprise SaaS?"
    ),
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=orchestrator.id)

messages = client.messages.list(thread_id=thread.id)
for msg in messages:
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(tc.text.value)

# Cleanup all agents
client.threads.delete(thread.id)
client.delete_agent(orchestrator.id)
client.delete_agent(legal_agent.id)
client.delete_agent(finance_agent.id)
```

### Gotchas

- The `description` is the primary signal the orchestrator uses to route tasks. Write it precisely.
- Sub-agents must be created (and their IDs obtained) before creating the orchestrator.
- `ConnectedAgentTool` has no `resources` — the connection is handled entirely through the definition.
- Each `ConnectedAgentTool` counts as one tool entry. You can have multiple connected agents in a single toolset.

---

## AgentEventHandler (Streaming)

`AgentEventHandler` is the base class for handling streaming agent responses. Override its event methods to process tokens, tool calls, and lifecycle events as they arrive, rather than waiting for the full response.

### Import

```python
from azure.ai.agents.models import AgentEventHandler
from azure.ai.agents.models import AsyncAgentEventHandler  # for async streaming
```

### Class signature

```python
class AgentEventHandler:
    def on_message_delta(self, delta: MessageDeltaChunk) -> Optional[T]: ...
    def on_thread_message(self, message: ThreadMessage) -> Optional[T]: ...
    def on_thread_run(self, run: ThreadRun) -> Optional[T]: ...
    def on_run_step(self, step: RunStep) -> Optional[T]: ...
    def on_run_step_delta(self, delta: RunStepDeltaChunk) -> Optional[T]: ...
    def on_error(self, data: str) -> Optional[T]: ...
    def on_done(self) -> Optional[T]: ...
    def submit_tool_outputs(
        self, run: ThreadRun, event_handler, allow_retry: bool
    ) -> None: ...
    def set_max_retry(self, max_retry: int) -> None: ...
```

### `MessageDeltaChunk` — text property

```python
class MessageDeltaChunk:
    @property
    def text(self) -> str:
        """Concatenated text from all delta content items."""
```

Use `delta.text` to access the incremental text token(s) in each streaming event.

### `AgentRunStream` context manager

```python
# Using until_done (blocks until stream completes)
with client.runs.stream(
    thread_id=thread.id,
    agent_id=agent.id,
    event_handler=handler,
) as stream:
    stream.until_done()

# Or iterate events manually
with client.runs.stream(thread_id=thread.id, agent_id=agent.id) as stream:
    for event_type, event_data, raw_response in stream:
        print(event_type, event_data)
```

### Example: Streaming agent with live output

```python
import os
import sys
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    AgentEventHandler,
    MessageDeltaChunk,
    ThreadMessage,
    ThreadRun,
    RunStep,
    MessageRole,
)
from azure.identity import DefaultAzureCredential


class LiveOutputHandler(AgentEventHandler):
    """Prints tokens to stdout as they arrive and reports lifecycle events."""

    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        # Print each token without a newline — creates live streaming effect
        sys.stdout.write(delta.text)
        sys.stdout.flush()

    def on_thread_message(self, message: ThreadMessage) -> None:
        if message.role == "assistant":
            # Called when a complete message object is available
            print(f"\n[Message complete: {message.id}]")

    def on_thread_run(self, run: ThreadRun) -> None:
        print(f"\n[Run status changed: {run.status}]")

    def on_run_step(self, step: RunStep) -> None:
        print(f"\n[Run step: {step.type} — {step.status}]")

    def on_error(self, data: str) -> None:
        print(f"\n[Error: {data}]")

    def on_done(self) -> None:
        print("\n[Stream complete]")


client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

agent = client.create_agent(
    model="gpt-4o",
    name="streaming-demo",
    instructions="You are a creative writer. Write vivid, engaging prose.",
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="Write a short paragraph describing a rainy afternoon in Edinburgh.",
)

handler = LiveOutputHandler()
with client.runs.stream(
    thread_id=thread.id,
    agent_id=agent.id,
    event_handler=handler,
) as stream:
    stream.until_done()

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

### Example: Streaming with tool calls

To handle tool calls during streaming, override `submit_tool_outputs`. The SDK calls this method automatically when the run reaches `REQUIRES_ACTION` state during streaming.

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    AgentEventHandler,
    MessageDeltaChunk,
    ThreadRun,
    FunctionTool,
    ToolSet,
    MessageRole,
    RunStatus,
)
from azure.identity import DefaultAzureCredential
import json


def get_current_time(timezone: str) -> str:
    """
    Get the current time in a specified timezone.

    :param timezone: Timezone identifier (e.g. 'Europe/London', 'US/Eastern').
    """
    from datetime import datetime
    import pytz
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    return f"Current time in {timezone}: {now.strftime('%H:%M:%S %Z')}"


class ToolAwareStreamHandler(AgentEventHandler):

    def __init__(self):
        super().__init__()
        self._text_buffer = []

    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        self._text_buffer.append(delta.text)
        print(delta.text, end="", flush=True)

    def on_thread_run(self, run: ThreadRun) -> None:
        if run.status == RunStatus.REQUIRES_ACTION:
            print("\n[Tool calls required — executing...]")

    def on_done(self) -> None:
        print("\n[Done]")
        print(f"Total characters received: {sum(len(t) for t in self._text_buffer)}")

    def on_error(self, data: str) -> None:
        print(f"\n[Stream error: {data}]")


client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

time_tool = FunctionTool(functions={get_current_time})
toolset = ToolSet()
toolset.add(time_tool)

agent = client.create_agent(
    model="gpt-4o",
    name="time-agent",
    instructions="Help users find the current time in different timezones.",
    toolset=toolset,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="What time is it right now in London and New York?",
)

handler = ToolAwareStreamHandler()
with client.runs.stream(
    thread_id=thread.id,
    agent_id=agent.id,
    event_handler=handler,
    toolset=toolset,   # Pass toolset to stream so it can auto-execute tools
) as stream:
    stream.until_done()

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

### Example: Async streaming

```python
import asyncio
import os
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import AsyncAgentEventHandler, MessageDeltaChunk, MessageRole
from azure.identity.aio import DefaultAzureCredential


class AsyncLiveHandler(AsyncAgentEventHandler):

    async def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        print(delta.text, end="", flush=True)

    async def on_done(self) -> None:
        print("\n[Async stream complete]")

    async def on_error(self, data: str) -> None:
        print(f"\n[Error: {data}]")


async def main() -> None:
    async with AgentsClient(
        endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
        credential=DefaultAzureCredential(),
    ) as client:
        agent = await client.create_agent(
            model="gpt-4o",
            name="async-streamer",
            instructions="Be concise and informative.",
        )
        thread = await client.threads.create()
        await client.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content="Explain what makes Python's asyncio special in two sentences.",
        )

        handler = AsyncLiveHandler()
        async with client.runs.stream(
            thread_id=thread.id,
            agent_id=agent.id,
            event_handler=handler,
        ) as stream:
            await stream.until_done()

        await client.threads.delete(thread.id)
        await client.delete_agent(agent.id)


asyncio.run(main())
```

### RunStatus enum values

| Value | Meaning |
|-------|---------|
| `QUEUED` | Run is queued, not yet started |
| `IN_PROGRESS` | Run is actively executing |
| `REQUIRES_ACTION` | Waiting for tool output submission |
| `CANCELLING` | Cancellation in progress |
| `CANCELLED` | Run was cancelled |
| `FAILED` | Run failed; inspect `run.last_error` |
| `COMPLETED` | Run finished successfully |
| `INCOMPLETE` | Run stopped due to token or time limits |
| `EXPIRED` | Run exceeded the expiry deadline |

---

## OpenApiTool

`OpenApiTool` enables agents to call external REST APIs described by an OpenAPI (Swagger) specification. The agent reads the spec to understand available endpoints and invokes them as needed.

### Import

```python
from azure.ai.agents.models import (
    OpenApiTool,
    OpenApiAnonymousAuthDetails,
    OpenApiConnectionAuthDetails,
    OpenApiManagedAuthDetails,
    OpenApiManagedSecurityScheme,
)
```

### Class signature

```python
class OpenApiTool:
    def __init__(
        self,
        name: str,
        description: str,
        spec: Any,                              # OpenAPI spec as dict or JSON string
        auth: OpenApiAuthDetails,
        default_parameters: Optional[List[str]] = None,
    )
    
    def add_definition(
        self,
        name: str,
        description: str,
        spec: Any,
        auth: OpenApiAuthDetails = None,
        default_parameters: Optional[List[str]] = None,
    ) -> None
    
    def remove_definition(self, name: str) -> None
    
    @property
    def definitions(self) -> List[OpenApiToolDefinition]
    
    @property
    def resources(self) -> ToolResources  # Empty
```

### Authentication types

| Class | Use case |
|-------|----------|
| `OpenApiAnonymousAuthDetails()` | Public APIs with no authentication |
| `OpenApiConnectionAuthDetails(connection_id="...")` | APIs authenticated via an Azure AI Foundry connection |
| `OpenApiManagedAuthDetails(security_scheme=OpenApiManagedSecurityScheme(audience="..."))` | APIs authenticated via managed identity |

### Example: Calling a public REST API

```python
import os
import json
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    OpenApiTool,
    OpenApiAnonymousAuthDetails,
    ToolSet,
    MessageRole,
)
from azure.identity import DefaultAzureCredential

# OpenAPI spec for a public weather API (simplified)
weather_spec = {
    "openapi": "3.0.0",
    "info": {
        "title": "Open-Meteo Weather API",
        "version": "1.0.0",
        "description": "Free weather API providing forecasts.",
    },
    "servers": [{"url": "https://api.open-meteo.com"}],
    "paths": {
        "/v1/forecast": {
            "get": {
                "operationId": "getWeatherForecast",
                "summary": "Get weather forecast for a location",
                "parameters": [
                    {
                        "name": "latitude",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "number"},
                        "description": "Latitude of the location.",
                    },
                    {
                        "name": "longitude",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "number"},
                        "description": "Longitude of the location.",
                    },
                    {
                        "name": "current_weather",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "boolean"},
                        "description": "Include current weather data.",
                    },
                ],
                "responses": {
                    "200": {
                        "description": "Successful forecast response",
                    }
                },
            }
        }
    },
}

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

weather_tool = OpenApiTool(
    name="weather_api",
    description="Provides real-time weather forecasts for any location by coordinates.",
    spec=weather_spec,
    auth=OpenApiAnonymousAuthDetails(),
)

toolset = ToolSet()
toolset.add(weather_tool)

agent = client.create_agent(
    model="gpt-4o",
    name="weather-openapi-agent",
    instructions=(
        "You are a weather assistant. Use the weather_api tool to fetch live "
        "forecast data. Edinburgh is at latitude 55.95, longitude -3.19."
    ),
    toolset=toolset,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="What is the current weather in Edinburgh?",
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

messages = client.messages.list(thread_id=thread.id)
for msg in messages:
    if msg.role == "assistant":
        for tc in msg.text_messages:
            print(tc.text.value)

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

### Example: Multiple API definitions with connection auth

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    OpenApiTool,
    OpenApiConnectionAuthDetails,
    ToolSet,
    MessageRole,
)
from azure.identity import DefaultAzureCredential

crm_spec = {...}    # Your CRM OpenAPI spec
billing_spec = {...} # Your billing system OpenAPI spec

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# Create the tool with the first API spec
api_tool = OpenApiTool(
    name="crm_api",
    description="CRM system for looking up customer accounts and contacts.",
    spec=crm_spec,
    auth=OpenApiConnectionAuthDetails(
        connection_id=os.environ["CRM_CONNECTION_ID"]
    ),
)

# Add a second API spec to the same tool object
api_tool.add_definition(
    name="billing_api",
    description="Billing system for querying invoices and payment history.",
    spec=billing_spec,
    auth=OpenApiConnectionAuthDetails(
        connection_id=os.environ["BILLING_CONNECTION_ID"]
    ),
)

toolset = ToolSet()
toolset.add(api_tool)

agent = client.create_agent(
    model="gpt-4o",
    name="crm-billing-agent",
    instructions=(
        "You assist account managers. Use the CRM to look up customers "
        "and the billing system to check their invoice status."
    ),
    toolset=toolset,
)
```

### Example: Managed identity authentication

```python
from azure.ai.agents.models import (
    OpenApiTool,
    OpenApiManagedAuthDetails,
    OpenApiManagedSecurityScheme,
)

# Use managed identity when running inside Azure (AKS, App Service, Functions, etc.)
tool = OpenApiTool(
    name="internal_api",
    description="Internal Azure-hosted REST API.",
    spec=my_spec,
    auth=OpenApiManagedAuthDetails(
        security_scheme=OpenApiManagedSecurityScheme(
            audience="https://your-api.azurewebsites.net"
        )
    ),
)
```

---

## Async Variants Summary

The SDK provides complete async support. Every synchronous class has an async counterpart and follows the same method signature, with `await` added where appropriate.

| Sync class | Async class | Notes |
|------------|-------------|-------|
| `AgentsClient` | `azure.ai.agents.aio.AgentsClient` | Use `async with` or `await client.close()` |
| `FunctionTool` | `AsyncFunctionTool` | `execute()` is async; supports both sync and async functions |
| `ToolSet` | `AsyncToolSet` | `execute_tool_calls()` is async |
| `AgentEventHandler` | `AsyncAgentEventHandler` | All `on_*` methods are async |
| `AgentRunStream` | async context manager | Use `async with` and `await stream.until_done()` |

### Pattern for async client lifecycle

```python
import asyncio
import os
from azure.ai.agents.aio import AgentsClient
from azure.identity.aio import DefaultAzureCredential


async def main() -> None:
    # Pattern 1: async context manager (recommended)
    async with AgentsClient(
        endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
        credential=DefaultAzureCredential(),
    ) as client:
        agent = await client.create_agent(model="gpt-4o", name="async-demo",
                                          instructions="Be helpful.")
        thread = await client.threads.create()
        # ... do work ...
        await client.threads.delete(thread.id)
        await client.delete_agent(agent.id)
    # client is automatically closed here


async def main_explicit_close() -> None:
    # Pattern 2: explicit close (useful when client lifetime spans multiple functions)
    credential = DefaultAzureCredential()
    client = AgentsClient(
        endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
        credential=credential,
    )
    try:
        agent = await client.create_agent(model="gpt-4o", name="explicit-demo",
                                          instructions="Be helpful.")
        # ... do work ...
        await client.delete_agent(agent.id)
    finally:
        await client.close()
        await credential.close()


asyncio.run(main())
```

---

## Common Patterns and Gotchas

### Pattern: Manual polling loop

When you need fine-grained control over run status — for example, to emit progress updates or handle specific intermediate states — use the manual polling approach instead of `create_and_process`:

```python
import time
from azure.ai.agents.models import RunStatus

# Start the run without polling
run = client.runs.create(thread_id=thread.id, agent_id=agent.id)

while run.status in (RunStatus.QUEUED, RunStatus.IN_PROGRESS, RunStatus.CANCELLING):
    time.sleep(1)
    run = client.runs.get(thread_id=thread.id, run_id=run.id)
    print(f"  Status: {run.status}")

if run.status == RunStatus.REQUIRES_ACTION:
    # Handle tool calls manually
    tool_outputs = []
    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
        # Execute the tool call and collect output
        output = my_function_dispatcher(tool_call)
        tool_outputs.append({
            "tool_call_id": tool_call.id,
            "output": str(output),
        })
    run = client.runs.submit_tool_outputs(
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=tool_outputs,
    )
elif run.status == RunStatus.FAILED:
    print(f"Run failed: {run.last_error}")
elif run.status == RunStatus.COMPLETED:
    print("Run completed successfully.")
```

### Pattern: Reusing threads for multi-turn conversations

Threads persist across multiple runs, making multi-turn conversations straightforward:

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import MessageRole
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

agent = client.create_agent(
    model="gpt-4o",
    name="conversational-agent",
    instructions="You are a helpful assistant with memory of previous messages.",
)

# Create thread once — reuse for all turns
thread = client.threads.create()

conversation = [
    "My name is Alice and I work in renewable energy.",
    "What sector do I work in?",          # Should recall 'renewable energy'
    "And what is my name?",               # Should recall 'Alice'
]

for user_input in conversation:
    client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=user_input,
    )
    run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
    
    # Get only the latest assistant message
    msgs = list(client.messages.list(thread_id=thread.id))
    latest_assistant = next(
        (m for m in msgs if m.role == "assistant"),
        None,
    )
    if latest_assistant:
        for tc in latest_assistant.text_messages:
            print(f"User: {user_input}")
            print(f"Agent: {tc.text.value}\n")
            break

client.threads.delete(thread.id)
client.delete_agent(agent.id)
```

### Pattern: Structured output with response_format

```python
from azure.ai.agents.models import AgentsResponseFormatJsonSchema

# Define a JSON schema for structured output
schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Brief summary"},
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["summary", "sentiment", "confidence"],
}

agent = client.create_agent(
    model="gpt-4o",
    name="structured-agent",
    instructions="Analyse text and return structured JSON.",
    response_format=AgentsResponseFormatJsonSchema(
        json_schema=AgentsResponseFormatJsonSchemaType(
            name="sentiment_analysis",
            schema=schema,
            strict=True,
        )
    ),
)
```

### Gotcha: Do not mix sync and async

A `FunctionTool` (sync) cannot be added to an `AsyncToolSet`, and `AsyncFunctionTool` cannot be added to a synchronous `ToolSet`. The SDK enforces this and will raise a `ValueError` at add-time:

```python
# WRONG — will raise ValueError
from azure.ai.agents.models import AsyncFunctionTool, ToolSet
bad_toolset = ToolSet()
bad_toolset.add(AsyncFunctionTool(functions={my_async_fn}))  # ValueError!

# CORRECT
from azure.ai.agents.models import AsyncFunctionTool, AsyncToolSet
good_toolset = AsyncToolSet()
good_toolset.add(AsyncFunctionTool(functions={my_async_fn}))  # OK
```

### Gotcha: Endpoint URL format

The endpoint must include both the AI services resource ID and the project path. A common mistake is using just the base AI services hostname:

```python
# WRONG
endpoint = "https://my-aiservices.cognitiveservices.azure.com"

# CORRECT
endpoint = "https://my-aiservices.services.ai.azure.com/api/projects/my-project"
```

### Gotcha: Function type annotations are required

The `FunctionTool` schema introspection relies on Python type annotations. Without them, the SDK cannot generate a valid JSON schema for the model:

```python
# WRONG — no type annotations, schema will be incomplete or incorrect
def bad_function(city, unit):
    """Get weather."""
    return "sunny"

# CORRECT — full type annotations
def good_function(city: str, unit: str = "celsius") -> str:
    """
    Get weather for a city.

    :param city: The city name.
    :param unit: Temperature unit, either 'celsius' or 'fahrenheit'.
    """
    return "sunny"
```

### Gotcha: Cleanup ordering for multi-agent setups

When using `ConnectedAgentTool`, delete the orchestrator agent before deleting sub-agents to avoid dangling references during deletion:

```python
# CORRECT ordering for multi-agent cleanup
client.threads.delete(thread.id)
client.delete_agent(orchestrator_agent.id)   # Delete orchestrator first
client.delete_agent(sub_agent_1.id)
client.delete_agent(sub_agent_2.id)
```

### Gotcha: Vector store processing time

Vector stores are not immediately ready after creation. Always poll the status before using them:

```python
import time

vector_store = client.vector_stores.create(
    name="my-docs",
    file_ids=[file.id for file in uploaded_files],
)

# Poll until ready
max_wait = 120  # seconds
elapsed = 0
while vector_store.status == "in_progress" and elapsed < max_wait:
    time.sleep(3)
    elapsed += 3
    vector_store = client.vector_stores.get(vector_store.id)

if vector_store.status != "completed":
    raise RuntimeError(f"Vector store failed with status: {vector_store.status}")

print(f"Vector store ready: {vector_store.id}")
```

---

*This documentation was compiled from direct inspection of the `azure-ai-agents` v1.1.0 package source code. All class signatures, method names, and property names reflect the actual library implementation.*

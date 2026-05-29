---
title: "Azure AI Agents SDK (Python) — Class Deep Dives Vol. 5"
description: "Source-verified deep dives into 10 classes from azure-ai-agents 1.1.0: Agent model, AgentThread, ToolOutput, VectorStoreFileBatch, VectorStoreFile, FileInfo, MessageDeltaChunk, RunStepDeltaChunk, SubmitToolOutputsAction, and AgentRunStream with AgentEventHandler overrides."
framework: microsoft-agent-framework
language: python
---

# Azure AI Agents SDK (Python) — Class Deep Dives Vol. 5

**Package:** `azure-ai-agents`  
**Version covered:** 1.1.0  
**Verified against:** installed package at `/usr/local/lib/python3.11/dist-packages/azure/ai/agents/`

This is the fifth volume of source-verified class deep dives for the `azure-ai-agents` Python SDK. Earlier volumes focused on the tool and orchestration layer. This volume covers the **data model classes** — the objects you read from and write to — plus the streaming plumbing that connects them in real time.

Earlier volumes:
- **[Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sdk_class_deep_dives/)** — `AgentsClient`, `FunctionTool`, `ToolSet`, `CodeInterpreterTool`, `FileSearchTool`, `BingGroundingTool`, `ConnectedAgentTool`, `AgentEventHandler`, `ThreadMessage`, `OpenApiTool`
- **[Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sdk_class_deep_dives_v3/)** — `AsyncFunctionTool`, `AzureFunctionTool`, `AzureAISearchTool`, `VectorStore`, `ThreadRun`, `RunStep`, `ResponseFormatJsonSchema`, `TruncationObject`, `MessageAttachment`, `AsyncAgentEventHandler`
- **[Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sdk_class_deep_dives_v4/)** — `AgentsClient` (auto function calls, create_thread_and_process_run), `FunctionTool` (error handling, dynamic registration), `ToolSet`, `CodeInterpreterTool`, `FileSearchTool` + `VectorStore`, `AzureAISearchTool`, `BingGroundingTool`, `ConnectedAgentTool`, `AgentEventHandler`, `AsyncToolSet` + `AsyncFunctionTool`

---

## Table of Contents

1. [`Agent` — the core agent model](#1-agent--the-core-agent-model)
2. [`AgentThread` — conversation thread model](#2-agentthread--conversation-thread-model)
3. [`ToolOutput` — submitting tool call results](#3-tooloutput--submitting-tool-call-results)
4. [`VectorStoreFileBatch` + `VectorStoreFileCount` — batch file upload tracking](#4-vectorstorefilebatch--vectorstorefilecount--batch-file-upload-tracking)
5. [`VectorStoreFile` + `VectorStoreFileError` — individual file in a vector store](#5-vectorstorefile--vectorstorefileerror--individual-file-in-a-vector-store)
6. [`FileInfo` — uploaded file metadata](#6-fileinfo--uploaded-file-metadata)
7. [`MessageDeltaChunk` — streaming message content](#7-messagedeltachunk--streaming-message-content)
8. [`RunStepDeltaChunk` — streaming run step progress](#8-runstepdeltachunk--streaming-run-step-progress)
9. [`SubmitToolOutputsAction` — required action for tool execution](#9-submittooloutputsaction--required-action-for-tool-execution)
10. [`AgentRunStream` + `AgentEventHandler` — streaming context manager and event dispatch](#10-agentrunstream--agenteventhandler--streaming-context-manager-and-event-dispatch)

---

## 1. `Agent` — the core agent model

**Source:** `azure/ai/agents/models/_models.py`

The `Agent` model is the object you get back from `client.create_agent()` or `client.get_agent()`. It is a read-only snapshot of the persisted agent configuration on Azure — not a live object.

### Signature

```python
class Agent(_Model):
    id: str                                         # API-assigned identifier ("asst_…")
    object: Literal["assistant"]                    # always "assistant" — auto-set
    created_at: datetime.datetime                   # unix timestamp, formatted on read
    name: str                                       # display name
    description: str                                # agent description
    model: str                                      # model deployment name, e.g. "gpt-4o"
    instructions: str                               # system prompt
    tools: List[ToolDefinition]                     # tool definitions attached at creation
    tool_resources: ToolResources                   # per-tool resource bindings
    temperature: float                              # 0-2; higher = more random
    top_p: float                                    # nucleus sampling; alter temperature OR top_p
    response_format: Optional[AgentsResponseFormatOption]  # None | "auto" | AgentsResponseFormat | ResponseFormatJsonSchemaType
    metadata: Dict[str, str]                        # up to 16 pairs; keys ≤64 chars, values ≤512 chars
```

### Key points

- `object` is always `"assistant"` — set automatically in `__init__`; never pass it manually.
- `temperature` and `top_p` both control sampling. Microsoft recommends altering **one** at a time, not both simultaneously.
- `metadata` has hard limits: 16 key/value pairs maximum; keys ≤ 64 chars; values ≤ 512 chars.
- You do **not** construct `Agent` directly in normal use — the client returns it from API calls. The constructor exists for deserialization and testing.
- To update an agent, call `client.update_agent(agent_id, ...)` — `Agent` is immutable.

### Example 1: read back the agent you just created

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    FunctionTool, ToolResources, MessageRole, Agent
)
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

def lookup_product_price(product_id: str) -> str:
    """Return the price of a product."""
    prices = {"SKU-001": "£19.99", "SKU-002": "£34.50"}
    return prices.get(product_id, "Price not found")

tool = FunctionTool({lookup_product_price})

agent: Agent = client.create_agent(
    model="gpt-4o",
    name="PricingAssistant",
    description="Answers product pricing questions.",
    instructions="You help customers find product prices. Always use the lookup tool.",
    tools=tool.definitions,
    temperature=0.2,       # precise, deterministic
    top_p=1.0,
    metadata={"team": "ecommerce", "version": "1.0"},
)

# Inspect the returned Agent model
print(f"Agent ID  : {agent.id}")
print(f"Model     : {agent.model}")
print(f"Tools     : {[t.type for t in agent.tools]}")
print(f"Temp      : {agent.temperature}")
print(f"Metadata  : {agent.metadata}")
```

### Example 2: list agents and filter by metadata

```python
# Iterate all agents in the project and find those tagged for a team
for agent in client.list_agents():
    team = agent.metadata.get("team")
    if team == "ecommerce":
        print(f"  {agent.name} ({agent.id}) — model: {agent.model}")
```

### Example 3: update model and instructions

```python
# Update an existing agent in-place (returns a new Agent snapshot)
updated: Agent = client.update_agent(
    agent_id=agent.id,
    model="gpt-4o-mini",                              # switch to cheaper model
    instructions=agent.instructions + "\nKeep replies under 50 words.",
    metadata={**agent.metadata, "version": "1.1"},
)
print(f"Updated to model: {updated.model}")
```

---

## 2. `AgentThread` — conversation thread model

**Source:** `azure/ai/agents/models/_models.py`

`AgentThread` represents a conversation thread. Threads are persistent server-side objects that store the message history. Each call to a run references a thread.

### Signature

```python
class AgentThread(_Model):
    id: str                        # "thread_…"
    object: Literal["thread"]      # always "thread" — auto-set
    created_at: datetime.datetime  # creation timestamp
    tool_resources: ToolResources  # thread-level resource overrides (file IDs, vector store IDs)
    metadata: Dict[str, str]       # up to 16 key/value pairs
```

### Key points

- Thread-level `tool_resources` **override** the agent-level ones for this thread. Use this to give different threads access to different files without creating different agents.
- Threads accumulate messages until deleted. Long-running threads may hit token limits; use `TruncationObject` on the run to manage context window size.
- Deleting a thread removes all its messages permanently.
- Like `Agent`, you normally get `AgentThread` back from the client rather than constructing it directly.

### Example 1: thread per user session with file search

```python
from azure.ai.agents.models import (
    AgentThread, ToolResources, FileSearchToolResource, AgentThreadCreationOptions
)

# Each user gets their own thread with their private vector store
def create_user_thread(client, user_id: str, vector_store_id: str) -> AgentThread:
    thread: AgentThread = client.threads.create(
        tool_resources=ToolResources(
            file_search=FileSearchToolResource(vector_store_ids=[vector_store_id])
        ),
        metadata={"user_id": user_id, "session_start": "2026-05-29"},
    )
    print(f"Thread {thread.id} created for user {thread.metadata['user_id']}")
    print(f"Vector stores: {thread.tool_resources.file_search.vector_store_ids}")
    return thread
```

### Example 2: thread per user with code interpreter files

```python
from azure.ai.agents.models import CodeInterpreterToolResource

def create_analysis_thread(client, file_ids: list[str]) -> AgentThread:
    thread = client.threads.create(
        tool_resources=ToolResources(
            code_interpreter=CodeInterpreterToolResource(file_ids=file_ids)
        ),
        metadata={"purpose": "data_analysis"},
    )
    return thread

# Example: upload two datasets, then run the agent
with open("sales_q1.csv", "rb") as f:
    file_q1 = client.upload_file(f, purpose="assistants")
with open("sales_q2.csv", "rb") as f:
    file_q2 = client.upload_file(f, purpose="assistants")

thread = create_analysis_thread(client, [file_q1.id, file_q2.id])
```

### Example 3: thread lifecycle — create, use, delete

```python
# Create thread with an initial message using AgentThreadCreationOptions
from azure.ai.agents.models import AgentThreadCreationOptions, ThreadMessageOptions

thread_options = AgentThreadCreationOptions(
    messages=[
        ThreadMessageOptions(
            role=MessageRole.USER,
            content="Summarise the sales data and highlight anomalies.",
        )
    ],
    metadata={"job_id": "batch-001"},
)

run = client.create_thread_and_process_run(
    agent_id=agent.id,
    thread=thread_options,
)
print(f"Thread {run.thread_id} completed: {run.status}")

# Clean up when done
client.threads.delete(run.thread_id)
```

---

## 3. `ToolOutput` — submitting tool call results

**Source:** `azure/ai/agents/models/_models.py`

`ToolOutput` is the payload you send back to the agent when it calls one of your function tools and the run reaches `requires_action`. Every tool call in `SubmitToolOutputsAction.submit_tool_outputs.tool_calls` must have a matching `ToolOutput` in the submission — omitting one leaves the run blocked.

### Signature

```python
class ToolOutput(_Model):
    tool_call_id: Optional[str]   # ID from RequiredFunctionToolCall.id
    output: Optional[str]         # Result as a string (JSON-encode complex types)
```

Both fields are technically optional in the constructor, but in practice **both are always required** — an output without an ID is ignored, and an ID without output tells the model the tool returned nothing.

### Key points

- `output` is always a plain string. Serialise dicts and lists with `json.dumps()`.
- `tool_call_id` must exactly match the `id` from the corresponding `RequiredFunctionToolCall` — a mismatch silently fails.
- Submit a list of `ToolOutput` objects; the number of items must equal the number of tool calls in the action.
- Error outputs are valid — pass `output=json.dumps({"error": "..."})` to signal failure gracefully.

### Example 1: manual tool dispatch loop

```python
import json
import time
from azure.ai.agents.models import ToolOutput, SubmitToolOutputsAction, RunStatus

def get_stock_price(symbol: str) -> dict:
    prices = {"AAPL": 182.50, "MSFT": 415.20, "GOOGL": 175.80}
    price = prices.get(symbol.upper())
    if price is None:
        return {"error": f"Unknown symbol: {symbol}"}
    return {"symbol": symbol.upper(), "price": price, "currency": "USD"}

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="What are the current prices of AAPL and MSFT?",
)

run = client.runs.create(thread_id=thread.id, agent_id=agent.id)

# Poll and handle required actions manually
while run.status in (RunStatus.QUEUED, RunStatus.IN_PROGRESS, RunStatus.REQUIRES_ACTION):
    if run.status == RunStatus.REQUIRES_ACTION:
        action = run.required_action
        if isinstance(action, SubmitToolOutputsAction):
            outputs = []
            for tool_call in action.submit_tool_outputs.tool_calls:
                if tool_call.type == "function":
                    args = json.loads(tool_call.function.arguments)
                    result = get_stock_price(**args)
                    outputs.append(
                        ToolOutput(
                            tool_call_id=tool_call.id,
                            output=json.dumps(result),
                        )
                    )
            run = client.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=outputs,
            )
    else:
        time.sleep(1)
        run = client.runs.get(thread_id=thread.id, run_id=run.id)

print(f"Run finished: {run.status}")
```

### Example 2: error-tolerant multi-tool dispatch

```python
def dispatch_tool_call(tool_call) -> ToolOutput:
    """Execute a single tool call; wrap exceptions as error outputs."""
    try:
        args = json.loads(tool_call.function.arguments)
        # Route by function name
        if tool_call.function.name == "get_stock_price":
            result = get_stock_price(**args)
        elif tool_call.function.name == "convert_currency":
            result = convert_currency(**args)
        else:
            result = {"error": f"Unknown function: {tool_call.function.name}"}
        return ToolOutput(tool_call_id=tool_call.id, output=json.dumps(result))
    except Exception as exc:
        return ToolOutput(tool_call_id=tool_call.id, output=json.dumps({"error": str(exc)}))

# Submit all outputs in one batch
if isinstance(run.required_action, SubmitToolOutputsAction):
    outputs = [
        dispatch_tool_call(tc)
        for tc in run.required_action.submit_tool_outputs.tool_calls
    ]
    run = client.runs.submit_tool_outputs(
        thread_id=thread.id, run_id=run.id, tool_outputs=outputs
    )
```

---

## 4. `VectorStoreFileBatch` + `VectorStoreFileCount` — batch file upload tracking

**Source:** `azure/ai/agents/models/_models.py`

When you upload multiple files to a vector store in one call, the SDK returns a `VectorStoreFileBatch` that tracks the aggregate processing status. Poll it until all files reach a terminal state.

### Signatures

```python
class VectorStoreFileBatch(_Model):
    id: str                                               # "vsfb_…"
    object: Literal["vector_store.files_batch"]           # auto-set
    created_at: datetime.datetime
    vector_store_id: str                                  # parent vector store
    status: Union[str, VectorStoreFileBatchStatus]        # "in_progress" | "completed" | "cancelled" | "failed"
    file_counts: VectorStoreFileCount

class VectorStoreFileCount(_Model):
    in_progress: int   # files still being embedded
    completed: int     # files ready for search
    failed: int        # files that failed (check individual VectorStoreFile for details)
    cancelled: int
    total: int         # sum of all above
```

### Key points

- Status progresses: `in_progress` → `completed` (or `cancelled`/`failed`).
- A batch can reach `completed` but still have individual file failures; check `file_counts.failed > 0` and inspect individual `VectorStoreFile` objects to see error details.
- Batches are immutable — poll for updates via `client.vector_stores.file_batches.get(...)`.
- `usage_bytes` is per-file (on `VectorStoreFile`), not on the batch itself.

### Example 1: batch upload with polling

```python
import time
from azure.ai.agents.models import VectorStoreFileBatch, VectorStoreFileBatchStatus

# Upload three files then create the batch
file_paths = ["report_q1.pdf", "report_q2.pdf", "report_q3.pdf"]
file_ids = []
for path in file_paths:
    with open(path, "rb") as f:
        info = client.upload_file(f, purpose="assistants")
        file_ids.append(info.id)
        print(f"Uploaded {path} → {info.id}")

# Create a vector store and add files as a batch
vector_store = client.vector_stores.create(name="quarterly-reports")
batch: VectorStoreFileBatch = client.vector_stores.file_batches.create_and_poll(
    vector_store_id=vector_store.id,
    file_ids=file_ids,
)

print(f"Batch {batch.id}: {batch.status}")
print(f"  Completed : {batch.file_counts.completed}/{batch.file_counts.total}")
print(f"  Failed    : {batch.file_counts.failed}")
```

### Example 2: manual poll loop with progress reporting

```python
def upload_and_watch(client, vector_store_id: str, file_ids: list[str]) -> VectorStoreFileBatch:
    batch = client.vector_stores.file_batches.create(
        vector_store_id=vector_store_id,
        file_ids=file_ids,
    )

    while batch.status == VectorStoreFileBatchStatus.IN_PROGRESS:
        counts = batch.file_counts
        pct = int(100 * (counts.completed + counts.failed) / counts.total) if counts.total else 0
        print(f"  [{pct:3d}%] done={counts.completed} failed={counts.failed} pending={counts.in_progress}")
        time.sleep(2)
        batch = client.vector_stores.file_batches.get(
            vector_store_id=vector_store_id,
            file_batch_id=batch.id,
        )

    if batch.file_counts.failed > 0:
        print(f"WARNING: {batch.file_counts.failed} file(s) failed embedding.")
    return batch
```

---

## 5. `VectorStoreFile` + `VectorStoreFileError` — individual file in a vector store

**Source:** `azure/ai/agents/models/_models.py`

`VectorStoreFile` represents a single file's embedding status within a vector store. Inspect it when a batch has failures or when you need per-file detail.

### Signatures

```python
class VectorStoreFile(_Model):
    id: str                                                    # matches the uploaded FileInfo.id
    object: Literal["vector_store.file"]                       # auto-set
    usage_bytes: int                                           # embedded size — may differ from raw file size
    created_at: datetime.datetime
    vector_store_id: str
    status: Union[str, VectorStoreFileStatus]                  # "in_progress" | "completed" | "failed" | "cancelled"
    last_error: VectorStoreFileError                           # None if no errors
    chunking_strategy: VectorStoreChunkingStrategyResponse     # how the file was split

class VectorStoreFileError(_Model):
    code: Union[str, VectorStoreFileErrorCode]   # "server_error" | "invalid_file" | "unsupported_file"
    message: str                                  # human-readable description
```

### Key points

- `usage_bytes` reflects the storage consumed by the embeddings, not the raw file. PDFs typically expand significantly.
- `last_error` is always present as an attribute but may be `None` when there are no errors.
- A file with `status == "failed"` and `last_error.code == "unsupported_file"` means the file type cannot be embedded (e.g., an image-only PDF). Consider converting to plain text before uploading.
- The `chunking_strategy` on the file shows what was actually applied (vs. what was requested, which might have been `auto`).

### Example 1: inspect all files in a vector store

```python
from azure.ai.agents.models import VectorStoreFile, VectorStoreFileStatus

def audit_vector_store(client, vector_store_id: str):
    """List all files and report on any failures."""
    for vs_file in client.vector_stores.files.list(vector_store_id=vector_store_id):
        vs_file: VectorStoreFile
        status_icon = "✓" if vs_file.status == VectorStoreFileStatus.COMPLETED else "✗"
        print(f"  {status_icon} {vs_file.id}  ({vs_file.usage_bytes:,} bytes)  {vs_file.status}")
        if vs_file.status == VectorStoreFileStatus.FAILED and vs_file.last_error:
            print(f"     Error [{vs_file.last_error.code}]: {vs_file.last_error.message}")
```

### Example 2: retry failed files with a conversion step

```python
import pathlib

def retry_failed_files(client, vector_store_id: str):
    failed = [
        f for f in client.vector_stores.files.list(vector_store_id=vector_store_id)
        if f.status == "failed"
    ]
    print(f"Found {len(failed)} failed file(s) — attempting conversion and re-upload")

    new_ids = []
    for vs_file in failed:
        # Remove the failed file from the vector store
        client.vector_stores.files.delete(
            vector_store_id=vector_store_id,
            file_id=vs_file.id,
        )

        # Try converting the original (assumes you have a local copy)
        original_path = pathlib.Path(f"originals/{vs_file.id}.pdf")
        converted_path = convert_pdf_to_text(original_path)   # your conversion helper

        with open(converted_path, "rb") as f:
            new_file = client.upload_file(f, purpose="assistants")
        new_ids.append(new_file.id)
        print(f"  Re-uploaded {original_path.name} → {new_file.id}")

    # Re-add as a batch
    if new_ids:
        batch = client.vector_stores.file_batches.create_and_poll(
            vector_store_id=vector_store_id,
            file_ids=new_ids,
        )
        print(f"Re-upload batch {batch.status}: {batch.file_counts.completed}/{batch.file_counts.total} succeeded")
```

---

## 6. `FileInfo` — uploaded file metadata

**Source:** `azure/ai/agents/models/_models.py`

`FileInfo` is the object returned when you call `client.upload_file(...)` or `client.get_file(...)`. It carries the identity and status of an uploaded file.

### Signature

```python
class FileInfo(_Model):
    object: Literal["file"]                      # always "file" — auto-set
    id: str                                      # "file_…" — use this in tool_resources
    bytes: int                                   # raw upload size (not embedding size)
    filename: str                                # original filename
    created_at: datetime.datetime
    purpose: Union[str, FilePurpose]             # "assistants" | "assistants_output" | "vision"
    status: Optional[Union[str, FileState]]      # Azure OpenAI only; may be None
    status_details: Optional[str]               # error message if status == "error"
```

### `FilePurpose` values

| Value | Use |
|---|---|
| `"assistants"` | Code interpreter input files, file search source documents |
| `"assistants_output"` | Files produced by code interpreter — read-only |
| `"vision"` | Image files passed as vision input to the model |

### `FileState` values (Azure OpenAI only)

`"uploaded"` → `"pending"` → `"running"` → `"processed"` (or `"error"` → `"deleting"` → `"deleted"`)

### Key points

- `bytes` reflects the raw upload size, not the embedding footprint (which you find on `VectorStoreFile.usage_bytes`).
- On Azure AI Foundry endpoints, `status` is often `None` — do not gate on it unless you know you are using Azure OpenAI.
- Files with `purpose="assistants_output"` are created by code interpreter automatically; they are read-only.
- `id` is what you pass to `CodeInterpreterToolResource(file_ids=[...])` or `FileSearchTool`.

### Example 1: upload with purpose routing

```python
from azure.ai.agents.models import FileInfo, FilePurpose

def upload_dataset(client, local_path: str) -> FileInfo:
    """Upload a file for use with code interpreter."""
    with open(local_path, "rb") as f:
        file_info: FileInfo = client.upload_file(f, purpose=FilePurpose.ASSISTANTS)

    print(f"Uploaded: {file_info.filename}")
    print(f"  ID      : {file_info.id}")
    print(f"  Size    : {file_info.bytes:,} bytes")
    print(f"  Purpose : {file_info.purpose}")
    if file_info.status:
        print(f"  Status  : {file_info.status}")
    return file_info

file_info = upload_dataset(client, "monthly_metrics.xlsx")
```

### Example 2: clean up uploaded files by listing and deleting

```python
from azure.ai.agents.models import FilePurpose

def clean_up_old_assistant_files(client, keep_ids: set[str]):
    """Delete all 'assistants' files that are not in keep_ids."""
    all_files = client.list_files(purpose=FilePurpose.ASSISTANTS)
    deleted = 0
    for file_info in all_files:
        if file_info.id not in keep_ids:
            client.delete_file(file_info.id)
            deleted += 1
            print(f"Deleted {file_info.filename} ({file_info.id})")
    print(f"Removed {deleted} stale file(s)")
```

### Example 3: retrieve a code interpreter output file

```python
# After a run completes, code interpreter may produce output files
for message in client.messages.list(thread_id=thread.id):
    for content in message.content:
        # ImageFileContent has .image_file.file_id
        if hasattr(content, "image_file"):
            file_id = content.image_file.file_id
            file_info: FileInfo = client.get_file(file_id)
            print(f"Output image: {file_info.filename} ({file_info.bytes} bytes)")
            # Download it
            data = client.get_file_content(file_id)
            with open(f"output_{file_info.filename}", "wb") as f:
                for chunk in data:
                    f.write(chunk)
```

---

## 7. `MessageDeltaChunk` — streaming message content

**Source:** `azure/ai/agents/models/_patch.py` (subclass of `_models.py` generated class)

`MessageDeltaChunk` carries an incremental fragment of assistant message content during a streaming run. It is emitted on the `AgentStreamEvent.THREAD_MESSAGE_DELTA` event.

### Signature

```python
# Generated base (in _models.py):
class MessageDeltaChunkGenerated(_Model):
    id: str                              # message ID this delta belongs to
    object: Literal["thread.message.delta"]   # auto-set
    delta: MessageDelta                  # the changed fields

class MessageDelta(_Model):
    role: Optional[Union[str, MessageRole]]       # "assistant" for streaming responses
    content: List[MessageDeltaContent]            # list of incremental content items

# Patched subclass (in _patch.py) adds a convenience property:
class MessageDeltaChunk(MessageDeltaChunkGenerated):
    @property
    def text(self) -> str:
        """Concatenate all text value fragments in this delta into a single string."""
        if not self.delta or not self.delta.content:
            return ""
        return "".join(
            part.text.value or ""
            for part in self.delta.content
            if isinstance(part, MessageDeltaTextContent) and part.text
        )
```

### `MessageDeltaContent` subtypes

| Type | When used |
|---|---|
| `MessageDeltaTextContent` | Text fragments from the model; `.text.value` has the text; `.text.annotations` has citation deltas |
| `MessageDeltaImageFileContent` | Image files produced by code interpreter during streaming; `.image_file.file_id` |

### Key points

- Use `delta_chunk.text` (the patched convenience property) for the common case of text-only streaming — it concatenates all text parts in this chunk.
- For multi-modal or annotation-aware streaming, iterate `delta_chunk.delta.content` directly and check `isinstance(part, MessageDeltaTextContent)`.
- Each delta chunk represents a fragment, not a complete message — you must accumulate them across the stream to reconstruct the full message.
- `delta.content` items have an `index` field that identifies which content part of the message they belong to (in case of multi-part messages).

### Example 1: stream to stdout using the `.text` shortcut

```python
from azure.ai.agents.models import (
    AgentEventHandler, MessageDeltaChunk, ThreadRun, RunStatus
)

class PrintingHandler(AgentEventHandler):
    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        print(delta.text, end="", flush=True)   # convenience property from the patch

    def on_thread_run(self, run: ThreadRun) -> None:
        if run.status == RunStatus.FAILED:
            print(f"\n[Run failed: {run.last_error}]")

with client.runs.stream(
    thread_id=thread.id,
    agent_id=agent.id,
    event_handler=PrintingHandler(),
) as handler:
    for _event_type, _event_data, _result in handler:
        pass   # events are handled in the callbacks above

print()  # newline after streaming output
```

### Example 2: accumulate deltas and collect annotations

```python
from azure.ai.agents.models import (
    MessageDeltaChunk, MessageDeltaTextContent,
    MessageDeltaTextFileCitationAnnotation,
)

class AnnotationCollector(AgentEventHandler):
    def __init__(self):
        super().__init__()
        self._texts: dict[int, str] = {}        # index → accumulated text
        self._annotations: list = []

    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        for part in delta.delta.content:
            if isinstance(part, MessageDeltaTextContent):
                idx = part.index
                self._texts[idx] = self._texts.get(idx, "") + (part.text.value or "")
                for ann in (part.text.annotations or []):
                    if isinstance(ann, MessageDeltaTextFileCitationAnnotation):
                        self._annotations.append({
                            "text": ann.text,
                            "file_id": ann.file_citation.file_id if ann.file_citation else None,
                        })

    @property
    def full_text(self) -> str:
        return "".join(self._texts[i] for i in sorted(self._texts))

collector = AnnotationCollector()
with client.runs.stream(
    thread_id=thread.id, agent_id=agent.id, event_handler=collector
) as handler:
    for _ in handler:
        pass

print(collector.full_text)
print("Citations:", collector._annotations)
```

---

## 8. `RunStepDeltaChunk` — streaming run step progress

**Source:** `azure/ai/agents/models/_models.py`

`RunStepDeltaChunk` is emitted on `AgentStreamEvent.THREAD_RUN_STEP_DELTA` events. It shows incremental changes to a run step — most importantly, the tool calls being executed and their streaming outputs.

### Signature

```python
class RunStepDeltaChunk(_Model):
    id: str                                   # step ID
    object: Literal["thread.run.step.delta"]  # auto-set
    delta: RunStepDelta

class RunStepDelta(_Model):
    step_details: Optional[RunStepDeltaDetail]

# RunStepDeltaDetail is a discriminated union:
#   RunStepDeltaMessageCreationObject  — for message_creation steps
#   RunStepDeltaToolCallObject         — for tool_calls steps (code_interpreter, function, file_search, etc.)
```

### `RunStepDeltaToolCallObject` subtypes (inner `tool_calls` list)

| Type | Fields |
|---|---|
| `RunStepDeltaCodeInterpreterToolCall` | `.code_interpreter.input` (code being written), `.code_interpreter.outputs` (log/image deltas) |
| `RunStepDeltaFunctionToolCall` | `.function.name`, `.function.arguments` (streamed JSON), `.function.output` |
| `RunStepDeltaFileSearchToolCall` | `.file_search` (rarely emitted) |

### Key points

- Code interpreter input is streamed token-by-token via `RunStepDeltaCodeInterpreterToolCall.code_interpreter.input`.
- Function tool arguments are also streamed — accumulate them before parsing with `json.loads()`.
- Step IDs are stable within a run: multiple deltas for the same step share the same `id`.
- The step type (tool_calls vs. message_creation) is implicit in the `step_details` subtype.

### Example 1: live code interpreter output

```python
from azure.ai.agents.models import (
    AgentEventHandler, RunStepDeltaChunk, RunStep,
    RunStepDeltaCodeInterpreterToolCall, RunStepCodeInterpreterLogOutput,
)

class CodeStreamingHandler(AgentEventHandler):
    def on_run_step_delta(self, delta: RunStepDeltaChunk) -> None:
        details = delta.delta.step_details
        if details is None:
            return
        # Iterate each tool call delta in this step
        tool_calls = getattr(details, "tool_calls", None) or []
        for tc_delta in tool_calls:
            if isinstance(tc_delta, RunStepDeltaCodeInterpreterToolCall):
                ci = tc_delta.code_interpreter
                if ci:
                    # Stream the code being written by the model
                    if ci.input:
                        print(ci.input, end="", flush=True)
                    # Stream log output from execution
                    for output in (ci.outputs or []):
                        if isinstance(output, RunStepCodeInterpreterLogOutput):
                            print(f"\n[LOG] {output.logs}", flush=True)

    def on_run_step(self, step: RunStep) -> None:
        print(f"\n[Step {step.id}] {step.type} → {step.status}")
```

### Example 2: track tool call argument streaming

```python
import json
from azure.ai.agents.models import RunStepDeltaFunctionToolCall

class ToolCallTracker(AgentEventHandler):
    def __init__(self):
        super().__init__()
        self._call_args: dict[str, str] = {}     # tool_call_id → accumulated JSON string

    def on_run_step_delta(self, delta: RunStepDeltaChunk) -> None:
        details = delta.delta.step_details
        if not details:
            return
        for tc_delta in getattr(details, "tool_calls", []):
            if isinstance(tc_delta, RunStepDeltaFunctionToolCall):
                call_id = tc_delta.id
                if tc_delta.function and tc_delta.function.arguments:
                    self._call_args[call_id] = (
                        self._call_args.get(call_id, "") + tc_delta.function.arguments
                    )

    def get_args(self, call_id: str) -> dict:
        raw = self._call_args.get(call_id, "{}")
        return json.loads(raw)
```

---

## 9. `SubmitToolOutputsAction` — required action for tool execution

**Source:** `azure/ai/agents/models/_models.py`

When an agent's run reaches `requires_action` status and needs you to execute function tools, the run's `required_action` field holds a `SubmitToolOutputsAction`. This is the trigger for your tool-execution logic.

### Signature

```python
class SubmitToolOutputsAction(RequiredAction, discriminator="submit_tool_outputs"):
    type: Literal["submit_tool_outputs"]     # always "submit_tool_outputs" — auto-set
    submit_tool_outputs: SubmitToolOutputsDetails

class SubmitToolOutputsDetails(_Model):
    tool_calls: List[RequiredToolCall]       # each item is a RequiredFunctionToolCall in practice

class RequiredFunctionToolCall(RequiredToolCall):
    id: str                                  # matches ToolOutput.tool_call_id
    type: Literal["function"]                # discriminator
    function: RequiredFunctionToolCallDetails

class RequiredFunctionToolCallDetails(_Model):
    name: str            # function name to call
    arguments: str       # JSON-encoded arguments string (not yet parsed)
```

### Key points

- Check `isinstance(run.required_action, SubmitToolOutputsAction)` before accessing `.submit_tool_outputs`.
- Each `RequiredFunctionToolCall.function.arguments` is a **JSON string** — call `json.loads(tc.function.arguments)` before passing to your function.
- You must submit **all** tool call outputs in a single `runs.submit_tool_outputs()` call. Partial submission is not supported.
- If you use `enable_auto_function_calls()`, the SDK handles this dispatch automatically — you only need `SubmitToolOutputsAction` when implementing a manual loop or custom streaming handler.

### Example 1: complete manual tool dispatch

```python
import json
from azure.ai.agents.models import (
    SubmitToolOutputsAction, RequiredFunctionToolCall, ToolOutput, RunStatus
)

# Tool registry
TOOLS = {
    "get_weather": lambda city: f"{city}: 18°C, partly cloudy",
    "get_exchange_rate": lambda from_ccy, to_ccy: f"1 {from_ccy} = 0.92 {to_ccy}",
    "lookup_product": lambda sku: {"sku": sku, "name": "Widget Pro", "price": 29.99},
}

def execute_run_to_completion(client, thread_id: str, agent_id: str) -> str:
    run = client.runs.create(thread_id=thread_id, agent_id=agent_id)

    while run.status not in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.EXPIRED):
        if run.status == RunStatus.REQUIRES_ACTION:
            action = run.required_action
            if not isinstance(action, SubmitToolOutputsAction):
                break

            outputs = []
            for tc in action.submit_tool_outputs.tool_calls:
                if not isinstance(tc, RequiredFunctionToolCall):
                    continue
                fn = TOOLS.get(tc.function.name)
                if fn is None:
                    result = {"error": f"Function {tc.function.name!r} not registered"}
                else:
                    try:
                        args = json.loads(tc.function.arguments)
                        result = fn(**args)
                    except Exception as exc:
                        result = {"error": str(exc)}

                outputs.append(ToolOutput(
                    tool_call_id=tc.id,
                    output=json.dumps(result) if not isinstance(result, str) else result,
                ))

            run = client.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=outputs,
            )
        else:
            import time; time.sleep(1)
            run = client.runs.get(thread_id=thread_id, run_id=run.id)

    return run.status
```

### Example 2: streaming with manual tool submission

```python
from azure.ai.agents.models import (
    AgentEventHandler, ThreadRun, SubmitToolOutputsAction, ToolOutput
)
import json, time

class ManualToolHandler(AgentEventHandler):
    """Handler that streams output but executes tools manually."""

    def __init__(self, tool_registry: dict):
        super().__init__()
        self._tools = tool_registry

    def on_message_delta(self, delta) -> None:
        print(delta.text, end="", flush=True)

    def on_thread_run(self, run: ThreadRun) -> list[ToolOutput] | None:
        if (
            run.status == "requires_action"
            and isinstance(run.required_action, SubmitToolOutputsAction)
        ):
            outputs = []
            for tc in run.required_action.submit_tool_outputs.tool_calls:
                fn = self._tools.get(tc.function.name)
                args = json.loads(tc.function.arguments)
                result = fn(**args) if fn else {"error": "Not found"}
                outputs.append(ToolOutput(
                    tool_call_id=tc.id,
                    output=json.dumps(result),
                ))
            return outputs   # returned to AgentRunStream.submit_tool_outputs
        return None

with client.runs.stream(
    thread_id=thread.id,
    agent_id=agent.id,
    event_handler=ManualToolHandler(TOOLS),
) as handler:
    for _ in handler:
        pass
print()
```

---

## 10. `AgentRunStream` + `AgentEventHandler` — streaming context manager and event dispatch

**Source:** `azure/ai/agents/models/_patch.py`

`AgentRunStream` is the context manager returned by `client.runs.stream()`. It wires the HTTP response iterator to an `AgentEventHandler` (or your subclass) and manages resource lifecycle. `AgentEventHandler` is the base event dispatch class — subclass it to override any combination of its seven event hooks.

### `AgentRunStream` signature

```python
class AgentRunStream(Generic[BaseAgentEventHandlerT]):
    def __init__(
        self,
        response_iterator: Iterator[bytes],         # raw SSE byte stream
        submit_tool_outputs: Callable[
            [ThreadRun, BaseAgentEventHandlerT, bool], Any
        ],                                           # SDK-wired tool submission callback
        event_handler: BaseAgentEventHandlerT,       # your handler instance
    ):
        # Calls event_handler.initialize(response_iterator, submit_tool_outputs)
        ...

    def __enter__(self) -> BaseAgentEventHandlerT:   # returns the handler, not the stream
        return self.event_handler

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calls response_iterator.close() if it has that method
        ...
```

### `AgentEventHandler` overridable hooks

| Method | Called when | Argument type |
|---|---|---|
| `on_message_delta` | `THREAD_MESSAGE_DELTA` | `MessageDeltaChunk` |
| `on_thread_message` | `THREAD_MESSAGE_CREATED/COMPLETED/INCOMPLETE/IN_PROGRESS` | `ThreadMessage` |
| `on_thread_run` | Any `thread.run.*` event | `ThreadRun` |
| `on_run_step` | Any `thread.run.step.*` except delta | `RunStep` |
| `on_run_step_delta` | `THREAD_RUN_STEP_DELTA` | `RunStepDeltaChunk` |
| `on_error` | `ERROR` | `str` (raw data) |
| `on_done` | `DONE` | — |
| `on_unhandled_event` | Any unrecognised event type | `str, str` |

### Key points

- `__enter__` returns the **handler**, not the stream — pattern is:
  ```python
  with client.runs.stream(..., event_handler=MyHandler()) as handler:
      for event_type, event_data, result in handler:
          ...
  ```
- If `on_thread_run` returns a non-empty list of `ToolOutput`, `AgentEventHandler._process_event` uses it as the tool submission result. This is the hook for custom streaming tool dispatch.
- `set_max_retry(n)` controls how many times the SDK retries tool output submission when outputs contain errors (default: 10).
- `__exit__` closes the HTTP connection — always use `AgentRunStream` as a context manager to avoid connection leaks.
- The async version is `AsyncAgentRunStream` / `AsyncAgentEventHandler` in `azure.ai.agents.aio`.

### Example 1: minimal streaming with status tracking

```python
from azure.ai.agents.models import (
    AgentEventHandler, MessageDeltaChunk, ThreadRun, RunStep, RunStatus
)

class StatusTrackingHandler(AgentEventHandler):
    def __init__(self):
        super().__init__()
        self.run_status: str | None = None
        self.step_count: int = 0
        self.token_count: int = 0

    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        print(delta.text, end="", flush=True)

    def on_thread_run(self, run: ThreadRun) -> None:
        self.run_status = run.status
        if run.status == RunStatus.COMPLETED and run.usage:
            self.token_count = run.usage.total_tokens

    def on_run_step(self, step: RunStep) -> None:
        if step.status == "completed":
            self.step_count += 1

    def on_done(self) -> None:
        print(f"\n[Done] steps={self.step_count}, tokens={self.token_count}")

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role=MessageRole.USER,
    content="Explain the Azure AI Agents SDK in three bullet points.",
)

handler = StatusTrackingHandler()
with client.runs.stream(
    thread_id=thread.id,
    agent_id=agent.id,
    event_handler=handler,
) as stream_handler:
    for _ in stream_handler:
        pass

print(f"Run status: {handler.run_status}")
```

### Example 2: async streaming with `AsyncAgentRunStream`

```python
import asyncio
from azure.ai.agents.aio import AgentsClient as AsyncAgentsClient
from azure.ai.agents.models import AsyncAgentEventHandler, MessageDeltaChunk, ThreadRun

class AsyncPrintHandler(AsyncAgentEventHandler):
    async def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        print(delta.text, end="", flush=True)

    async def on_thread_run(self, run: ThreadRun) -> None:
        if run.status == "failed":
            print(f"\n[FAILED] {run.last_error}")

async def main():
    async with AsyncAgentsClient(
        endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
        credential=DefaultAzureCredential(),
    ) as async_client:
        agent = await async_client.create_agent(
            model="gpt-4o",
            name="AsyncAssistant",
            instructions="You are a concise assistant.",
        )
        thread = await async_client.threads.create()
        await async_client.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content="Name three Azure AI services.",
        )

        async with await async_client.runs.stream(
            thread_id=thread.id,
            agent_id=agent.id,
            event_handler=AsyncPrintHandler(),
        ) as handler:
            async for _ in handler:
                pass

        print()
        await async_client.threads.delete(thread.id)
        await async_client.delete_agent(agent.id)

asyncio.run(main())
```

### Example 3: combined streaming handler with tool execution and telemetry

```python
import json, time
from azure.ai.agents.models import (
    AgentEventHandler, MessageDeltaChunk, ThreadRun, RunStep,
    RunStepDeltaChunk, SubmitToolOutputsAction, ToolOutput,
)

class ProductionStreamHandler(AgentEventHandler):
    """Production-grade handler: streams output, executes tools, collects metrics."""

    def __init__(self, tool_registry: dict, trace_id: str):
        super().__init__()
        self._tools = tool_registry
        self._trace_id = trace_id
        self._text_parts: list[str] = []
        self._step_timings: dict[str, float] = {}
        self.set_max_retry(3)

    # ── text streaming ────────────────────────────────────────────────────────
    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        fragment = delta.text
        if fragment:
            self._text_parts.append(fragment)
            print(fragment, end="", flush=True)

    # ── run lifecycle ─────────────────────────────────────────────────────────
    def on_thread_run(self, run: ThreadRun) -> list[ToolOutput] | None:
        if run.status == "requires_action" and isinstance(
            run.required_action, SubmitToolOutputsAction
        ):
            outputs = []
            for tc in run.required_action.submit_tool_outputs.tool_calls:
                fn = self._tools.get(tc.function.name)
                try:
                    args = json.loads(tc.function.arguments)
                    result = fn(**args) if fn else {"error": "unregistered"}
                except Exception as exc:
                    result = {"error": str(exc)}
                outputs.append(ToolOutput(
                    tool_call_id=tc.id,
                    output=json.dumps(result),
                ))
            return outputs

        if run.status == "failed":
            print(f"\n[{self._trace_id}] Run failed: {run.last_error}", flush=True)

        return None

    # ── step timing ───────────────────────────────────────────────────────────
    def on_run_step(self, step: RunStep) -> None:
        if step.status == "in_progress":
            self._step_timings[step.id] = time.monotonic()
        elif step.status in ("completed", "failed") and step.id in self._step_timings:
            elapsed = time.monotonic() - self._step_timings.pop(step.id)
            print(f"\n  [step {step.type} took {elapsed:.2f}s]", flush=True)

    # ── completion ────────────────────────────────────────────────────────────
    def on_done(self) -> None:
        full_text = "".join(self._text_parts)
        print(f"\n[{self._trace_id}] Stream complete ({len(full_text)} chars)")

    def on_error(self, data: str) -> None:
        print(f"\n[{self._trace_id}] SSE error: {data}", flush=True)

# Usage
TOOLS = {
    "get_weather": lambda city: f"{city}: 21°C, sunny",
    "lookup_price": lambda sku: {"sku": sku, "price": 49.99},
}

handler = ProductionStreamHandler(TOOLS, trace_id="req-42")
with client.runs.stream(
    thread_id=thread.id,
    agent_id=agent.id,
    event_handler=handler,
) as stream:
    for _ in stream:
        pass
print()
```

---

## Patterns combining multiple classes

### Upload → embed → search in one function

```python
import time
from azure.ai.agents.models import (
    FileInfo, VectorStore, VectorStoreFileBatch, FileSearchTool,
    FileSearchToolDefinition, ToolResources, FileSearchToolResource,
)

def build_searchable_knowledgebase(
    client, agent_id: str, document_paths: list[str]
) -> tuple[str, str]:
    """Upload documents, embed them, and return (vector_store_id, thread_id)."""

    # 1. Upload raw files
    file_ids: list[str] = []
    for path in document_paths:
        with open(path, "rb") as f:
            info: FileInfo = client.upload_file(f, purpose="assistants")
        file_ids.append(info.id)
        print(f"  Uploaded: {info.filename} ({info.id})")

    # 2. Create vector store and embed in batch
    vs: VectorStore = client.vector_stores.create(name="knowledgebase")
    batch: VectorStoreFileBatch = client.vector_stores.file_batches.create_and_poll(
        vector_store_id=vs.id,
        file_ids=file_ids,
    )
    print(f"  Batch done: {batch.file_counts.completed}/{batch.file_counts.total} embedded")
    if batch.file_counts.failed > 0:
        print(f"  WARNING: {batch.file_counts.failed} file(s) failed — check VectorStoreFile.last_error")

    # 3. Create a thread wired to this vector store
    thread = client.threads.create(
        tool_resources=ToolResources(
            file_search=FileSearchToolResource(vector_store_ids=[vs.id])
        )
    )
    return vs.id, thread.id
```

### Streaming run with file-search result attribution

```python
from azure.ai.agents.models import (
    AgentEventHandler, MessageDeltaChunk, ThreadMessage,
    MessageTextFileCitationAnnotation,
)

class CitationHandler(AgentEventHandler):
    def __init__(self):
        super().__init__()
        self._text = ""

    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        self._text += delta.text
        print(delta.text, end="", flush=True)

    def on_thread_message(self, message: ThreadMessage) -> None:
        if message.status == "completed":
            print("\n\n--- Sources ---")
            for content in message.content:
                if hasattr(content, "text"):
                    for ann in (content.text.annotations or []):
                        if isinstance(ann, MessageTextFileCitationAnnotation):
                            print(f"  [{ann.text}] → file {ann.file_citation.file_id}")
```

---

## Quick reference

| Class | `object` literal | Returned by | Constructed by user? |
|---|---|---|---|
| `Agent` | `"assistant"` | `create_agent`, `get_agent`, `list_agents` | No — deserialization only |
| `AgentThread` | `"thread"` | `threads.create`, `threads.get` | No |
| `ToolOutput` | — | — | **Yes** — build when submitting tool results |
| `VectorStoreFileBatch` | `"vector_store.files_batch"` | `file_batches.create`, `file_batches.get` | No |
| `VectorStoreFile` | `"vector_store.file"` | `vector_stores.files.list`, `files.get` | No |
| `FileInfo` | `"file"` | `upload_file`, `get_file`, `list_files` | No |
| `MessageDeltaChunk` | `"thread.message.delta"` | `THREAD_MESSAGE_DELTA` stream event | No |
| `RunStepDeltaChunk` | `"thread.run.step.delta"` | `THREAD_RUN_STEP_DELTA` stream event | No |
| `SubmitToolOutputsAction` | `"submit_tool_outputs"` | `run.required_action` when `status == "requires_action"` | No |
| `AgentRunStream` | — | `runs.stream(...)` | No — use as context manager |

---

## Revision history

| Date | Version | Summary |
|------|---------|---------|
| 2026-05-29 | azure-ai-agents 1.1.0 | Library installed (1.1.0) and source inspected at `/usr/local/lib/python3.11/dist-packages/azure/ai/agents/`. Ten classes deep-dived from source: `Agent` (all 12 fields; metadata limits; update pattern), `AgentThread` (thread-level `tool_resources` override; lifecycle), `ToolOutput` (both fields required in practice; JSON serialization; error outputs), `VectorStoreFileBatch` + `VectorStoreFileCount` (status progression; partial-failure pattern; manual poll loop), `VectorStoreFile` + `VectorStoreFileError` (error codes; `usage_bytes` vs raw size; retry-with-conversion), `FileInfo` (purpose routing; `FileState` availability per endpoint; output file download), `MessageDeltaChunk` (`.text` patch convenience property; annotation accumulation; `MessageDeltaContent` subtypes), `RunStepDeltaChunk` (code-interpreter input streaming; function argument accumulation; step ID stability), `SubmitToolOutputsAction` (discriminated union check; partial-submission prohibition; manual vs. auto dispatch), `AgentRunStream` + `AgentEventHandler` (context manager lifecycle; all 8 overridable hooks; `set_max_retry`; async variant). New guide `microsoft_agent_framework_python_sdk_class_deep_dives_v5.md` created. Index updated with Zero→Hero step 23, Jump-to-topic card, Reference card, and this revision entry. | Claude |

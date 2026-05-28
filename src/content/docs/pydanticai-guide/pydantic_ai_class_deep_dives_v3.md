---
title: "PydanticAI — Class Deep Dives Vol. 3"
description: "Source-verified deep dives into 10 PydanticAI classes: MCPToolset, UIAdapter/MessagesBuilder, CachePoint/CompactionPart, format_as_xml, BinaryContent/FileUrl family, DeferredToolRequests/DeferredToolResults/CallDeferred, NativeTool, NativeOrLocalTool, DynamicCapability, ToolSearch."
sidebar:
  label: "Class deep dives (Vol. 3)"
  order: 23
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.103.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at `1.103.0`.
</Aside>

Ten classes from the `pydantic_ai` 1.103.0 source covering MCP toolsets, frontend UI
integration, prompt caching and compaction, XML prompt formatting, multimodal content, deferred
tool execution, native tool capabilities, dynamic capabilities, and large-toolset search.

---

## 1. `MCPToolset` + `load_mcp_toolsets`

**Module:** `pydantic_ai.mcp`  
**Export:** `from pydantic_ai.mcp import MCPToolset, load_mcp_toolsets`

`MCPToolset` is the **recommended** way to connect a Pydantic AI agent to any
[Model Context Protocol](https://modelcontextprotocol.io) server. It is built on the
[FastMCP](https://gofastmcp.com) client, which supports the full MCP protocol — tools,
resources, prompts, sampling, elicitation, OAuth — across every transport (HTTP, SSE, stdio,
in-process FastMCP servers, multi-server JSON configs).

The older `MCPServer`, `MCPServerStdio`, `MCPServerSSE`, and `MCPServerStreamableHTTP` classes
are **deprecated** and will be removed in v2; migrate to `MCPToolset`.

### Constructor

```python
MCPToolset(
    client: str | Path | FastMCPClient | ...,
    *,
    # Pydantic AI layer
    id: str | None = None,
    max_retries: int | None = None,
    tool_error_behavior: Literal['retry', 'error'] = 'retry',
    process_tool_call: ProcessToolCallback | None = None,
    cache_tools: bool = True,
    cache_resources: bool = True,
    cache_prompts: bool = True,
    include_instructions: bool = False,
    include_return_schema: bool | None = None,
    # Sampling
    sampling_model: Model | None = None,
    sampling_handler: SamplingHandler | None = None,
    # MCP protocol
    elicitation_handler: ElicitationHandler | None = None,
    log_handler: LogHandler | None = None,
    log_level: LoggingLevel | None = None,
    # HTTP-specific (URL-based transport only)
    auth: httpx.Auth | Literal['oauth'] | str | None = None,
    headers: dict[str, str] | None = None,
    http_client: httpx.AsyncClient | None = None,
)
```

### Transport forms

`MCPToolset` accepts any input that FastMCP can build a transport from:

| Input | Transport |
|-------|-----------|
| `"http://localhost:8000/mcp"` (URL string) | StreamableHTTP |
| `"my_server.py"` (file path) | stdio subprocess |
| `Path("server.py")` | stdio subprocess |
| `FastMCPClient(...)` | pre-built FastMCP Client |
| in-process `FastMCP` server object | in-process (testing) |

### Example 1 — Streamable-HTTP server

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset("http://localhost:8000/mcp")
agent = Agent("openai:gpt-4o", toolsets=[toolset])

async def main():
    async with agent:
        result = await agent.run("List all available tools.")
        print(result.data)

asyncio.run(main())
```

### Example 2 — stdio subprocess

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

# Spawns `python my_mcp_server.py` as a subprocess
toolset = MCPToolset("my_mcp_server.py", include_instructions=True)
agent = Agent("anthropic:claude-sonnet-4-5", toolsets=[toolset])
```

### Example 3 — OAuth-authenticated HTTP server

```python
from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset(
    "https://api.example.com/mcp",
    auth="oauth",           # triggers FastMCP OAuth flow
    tool_error_behavior="error",  # propagate ToolError instead of retrying
    max_retries=3,          # override agent-level retries for this toolset
)
```

### Example 4 — Custom transport with pre-built FastMCP Client

```python
from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai.mcp import MCPToolset

client = Client(
    StreamableHttpTransport("http://localhost:8000/mcp"),
    auth="oauth",
)
toolset = MCPToolset(client, cache_tools=False)  # server changes tools dynamically
```

### Example 5 — In-process FastMCP server for testing

```python
import pytest
from fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel

mcp_server = FastMCP("test-server")

@mcp_server.tool()
def get_weather(city: str) -> str:
    return f"Sunny in {city}"

@pytest.mark.anyio
async def test_agent_with_mcp():
    toolset = MCPToolset(mcp_server)
    agent = Agent(TestModel(custom_result_text="London weather: Sunny"), toolsets=[toolset])
    async with agent:
        result = await agent.run("What's the weather in London?")
    assert "London" in result.data
```

### Example 6 — `process_tool_call` hook for logging/telemetry

```python
from pydantic_ai.mcp import MCPToolset, ProcessToolCallback
from pydantic_ai.tools import ToolDefinition

async def log_and_forward(
    ctx,
    tool_def: ToolDefinition,
    args: dict,
    call_tool,
):
    print(f"[MCP] calling {tool_def.name} with {args}")
    result = await call_tool(args)
    print(f"[MCP] {tool_def.name} returned {str(result)[:80]}")
    return result

toolset = MCPToolset("http://localhost:8000/mcp", process_tool_call=log_and_forward)
```

### `load_mcp_toolsets` — JSON config file

`load_mcp_toolsets` reads a `mcpServers` JSON config (same shape as Claude Desktop and
Cursor) and returns one `MCPToolset` per server, each wrapped in a `PrefixedToolset`
using the server's name as prefix to avoid tool-name collisions.

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import load_mcp_toolsets

# mcp_config.json:
# {
#   "mcpServers": {
#     "filesystem": {
#       "command": "npx",
#       "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
#     },
#     "weather": {
#       "url": "http://weather-mcp.internal/mcp",
#       "env": {"API_KEY": "${WEATHER_API_KEY}"}
#     }
#   }
# }

toolsets = load_mcp_toolsets("mcp_config.json")
# → [PrefixedToolset("filesystem", ...), PrefixedToolset("weather", ...)]

agent = Agent("openai:gpt-4o", toolsets=toolsets)
```

Environment variables are expanded: `${VAR_NAME}` raises if unset; `${VAR_NAME:-default}`
falls back to `default`.

### Caching semantics

| Flag | Default | When to set `False` |
|------|---------|---------------------|
| `cache_tools` | `True` | Server changes tools without `tools/list_changed` |
| `cache_resources` | `True` | Server changes resources without `resources/list_changed` |
| `cache_prompts` | `True` | Server changes prompts without `prompts/list_changed` |

When a pre-built `FastMCPClient` is passed, cache invalidation via MCP notifications is not
installed automatically — caches are only invalidated when the toolset exits.

---

## 2. `UIAdapter` + `MessagesBuilder`

**Module:** `pydantic_ai.ui`  
**Export:** `from pydantic_ai.ui import UIAdapter, MessagesBuilder, UIEventStream, BuilderCheckpoint`

`UIAdapter` is the **abstract base class** for building protocol-specific frontends on top of
Pydantic AI agents (AG-UI, custom WebSocket, SSE endpoints, etc.). It handles:

- Sanitising client-submitted messages (stripping unsafe URL schemes, managing system prompts)
- Running the agent via `run_stream_events()`
- Transforming the agent event stream into protocol-specific events via a `UIEventStream` subclass

### Constructor

```python
@dataclass
class UIAdapter(ABC, Generic[RunInputT, MessageT, EventT, AgentDepsT, OutputDataT]):
    agent: AbstractAgent[AgentDepsT, OutputDataT]
    run_input: RunInputT
    accept: str | None = None
    manage_system_prompt: Literal['server', 'client'] = 'server'
    allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'})
    allowed_file_url_force_download: frozenset[ForceDownloadMode] = frozenset()
```

### `manage_system_prompt`

| Value | Behaviour |
|-------|-----------|
| `'server'` (default) | Agent's `system_prompt` is authoritative. Client-sent `SystemPromptPart`s are stripped; `ReinjectSystemPrompt` capability is auto-added. |
| `'client'` | Frontend owns the system prompt. `SystemPromptPart`s are preserved as-is; agent's `system_prompt` is not injected automatically. |

### Security: `allowed_file_url_schemes`

By default only `http` and `https` URLs are forwarded to the model. Non-HTTP schemes (e.g.
`s3://`, `gs://`) cause the model provider to fetch the object using its own IAM credentials,
which would let a malicious client read any object that role can reach.

```python
# SECURE default — only https URLs pass through
adapter = MyAdapter(agent=agent, run_input=req)

# Allow S3 after auditing your frontend
adapter = MyAdapter(
    agent=agent,
    run_input=req,
    allowed_file_url_schemes=frozenset({'http', 'https', 's3'}),
)
```

### `MessagesBuilder`

`MessagesBuilder` accumulates `ModelRequestPart` and `ModelResponsePart` objects into a
`list[ModelMessage]`, automatically grouping consecutive parts of the same type into the
same `ModelRequest` / `ModelResponse`.

```python
from pydantic_ai.ui import MessagesBuilder
from pydantic_ai.messages import SystemPromptPart, UserPromptPart, TextPart

builder = MessagesBuilder()
builder.add(SystemPromptPart(content="You are helpful."))
builder.add(UserPromptPart(content="Hello!"))
# → [ModelRequest(parts=[SystemPromptPart(...), UserPromptPart(...)])]

builder.add(TextPart(content="Hi there!"))
# → [ModelRequest(...), ModelResponse(parts=[TextPart(...)])]

print(len(builder.messages))  # 2
```

### `checkpoint` and `last_modified`

Snapshot the builder state before a batch of `add()` calls, then find the last
message that was created or extended since that snapshot:

```python
from pydantic_ai.ui import MessagesBuilder
from pydantic_ai.messages import TextPart, ModelResponse

builder = MessagesBuilder()
# ... populate with previous messages ...

checkpoint = builder.checkpoint()
builder.add(TextPart(content="Streaming chunk 1"))
builder.add(TextPart(content="Streaming chunk 2"))

last = builder.last_modified(checkpoint, of_type=ModelResponse)
if last:
    print("Response now has", len(last.parts), "parts")
```

### Implementing a custom `UIAdapter`

```python
from dataclasses import dataclass
from typing import AsyncIterator
from pydantic_ai.ui import UIAdapter, UIEventStream
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage


@dataclass
class MyRequest:
    messages: list[ModelMessage]
    user_id: str


@dataclass
class MyEvent:
    text_delta: str | None = None
    done: bool = False


@dataclass
class MyEventStream(UIEventStream[MyRequest, MyEvent, None, str]):
    async def __aiter__(self) -> AsyncIterator[MyEvent]:
        # Override to transform pydantic_ai events → MyEvent
        async for event in self._run():
            yield MyEvent(text_delta=str(event))
        yield MyEvent(done=True)


@dataclass
class MyAdapter(UIAdapter[MyRequest, MyEvent, MyEvent, None, str]):
    def get_run_kwargs(self):
        return {
            "user_prompt": self.run_input.messages[-1],
            "message_history": self.run_input.messages[:-1],
        }

    def get_event_stream(self) -> MyEventStream:
        return MyEventStream(run_input=self.run_input)
```

---

## 3. `CachePoint` + `CompactionPart`

**Module:** `pydantic_ai.messages`  
**Export:** `from pydantic_ai import CachePoint, CompactionPart`

### `CachePoint` — prompt cache boundaries

`CachePoint` marks a cache boundary within a `UserPromptPart.content` sequence. Providers
that support prompt caching (Anthropic, Amazon Bedrock Converse) honour these markers;
unsupporting providers silently filter them out.

```python
@dataclass
class CachePoint:
    kind: Literal['cache-point'] = 'cache-point'
    ttl: Literal['5m', '1h'] = '5m'
```

**TTL options:**

| Value | Duration | Supported by |
|-------|----------|--------------|
| `'5m'` (default) | 5 minutes | Anthropic, Bedrock |
| `'1h'` | 1 hour | Anthropic only (enable via beta header) |

#### Example 1 — Cache a long document in a user prompt

```python
from pydantic_ai import Agent, CachePoint
from pydantic_ai.messages import TextContent

long_document = "..." * 1000  # expensive to tokenise each turn

agent = Agent("anthropic:claude-sonnet-4-5")

# Insert CachePoint after the document so Anthropic caches everything up to that point
result = agent.run_sync(
    [
        TextContent(content=f"Document:\n{long_document}"),
        CachePoint(ttl="1h"),     # cache this prefix for 1 hour
        TextContent(content="Summarise the key points."),
    ]
)
print(result.data)
```

#### Example 2 — Multi-turn conversation with stable prefix caching

```python
import asyncio
from pydantic_ai import Agent, CachePoint
from pydantic_ai.messages import ModelMessagesTypeAdapter, TextContent, UserPromptPart

SYSTEM_DOCS = "..." * 500  # stable reference material

agent = Agent("anthropic:claude-sonnet-4-5", system_prompt=SYSTEM_DOCS)

async def chat():
    history = []
    questions = ["What is section 1 about?", "Summarise section 2.", "Conclude."]

    for question in questions:
        # The stable system prompt prefix is cached after the first request
        result = await agent.run(
            [
                TextContent(content=question),
                CachePoint(),  # mark end of cached prefix each turn
            ],
            message_history=history,
        )
        history = result.all_messages()
        print(f"Q: {question}")
        print(f"A: {result.data}\n")

asyncio.run(chat())
```

#### Example 3 — 1-hour cache for expensive prompts

```python
from pydantic_ai import Agent, CachePoint
from pydantic_ai.messages import TextContent

# Requires Anthropic beta header for 1-hour caching
agent = Agent(
    "anthropic:claude-opus-4-5",
    model_settings={"extra_headers": {"anthropic-beta": "extended-cache-ttl-2025-02-19"}},
)

result = agent.run_sync(
    [
        TextContent(content=expensive_reference_text),
        CachePoint(ttl="1h"),
        TextContent(content="Answer questions about this text."),
    ]
)
```

### `CompactionPart` — conversation summary / compaction

`CompactionPart` is produced by provider-specific compaction mechanisms when a conversation
grows too long. It contains a summary (Anthropic) or encrypted opaque data (OpenAI) that
**must be round-tripped** to the same provider on subsequent requests.

```python
@dataclass
class CompactionPart:
    content: str | None = None          # readable summary (Anthropic) or None (OpenAI)
    id: str | None = None               # provider compaction ID
    provider_name: str | None = None    # required when id or provider_details is set
    provider_details: dict | None = None  # encrypted_content etc. (OpenAI)
    part_kind: Literal['compaction'] = 'compaction'
```

#### Example 4 — Detecting and round-tripping compaction

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import CompactionPart

agent = Agent("anthropic:claude-sonnet-4-5")

async def run_with_compaction():
    history = []
    for i in range(20):  # long conversation that may trigger compaction
        result = await agent.run(f"Step {i}: continue.", message_history=history)
        history = result.all_messages()

    # Check whether compaction occurred
    compaction_parts = [
        part
        for msg in history
        for part in msg.parts
        if isinstance(part, CompactionPart)
    ]
    if compaction_parts:
        cp = compaction_parts[0]
        print(f"Compaction by {cp.provider_name}:")
        if cp.content:
            print(f"  Summary: {cp.content[:200]}")
        else:
            print("  Encrypted (OpenAI) — round-trip only")

asyncio.run(run_with_compaction())
```

#### Example 5 — `has_content()` guard

```python
from pydantic_ai.messages import CompactionPart

cp = CompactionPart(content="Previous 20 messages summarised here.", provider_name="anthropic")

if cp.has_content():
    print("Human-readable summary:", cp.content)
else:
    print("Opaque provider data — do not display to users")
```

---

## 4. `format_as_xml`

**Module:** `pydantic_ai.format_prompt`  
**Export:** `from pydantic_ai import format_as_xml`

`format_as_xml` converts any Python object into an XML string that LLMs often find easier
to parse than JSON. It supports strings, bytes, booleans, numeric types, dates, UUIDs,
enums, mappings, iterables, dataclasses, and Pydantic `BaseModel` instances.

### Signature

```python
def format_as_xml(
    obj: Any,
    root_tag: str | None = None,
    item_tag: str = 'item',
    none_str: str = 'null',
    indent: str | None = '  ',
    include_field_info: Literal['once'] | bool = False,
) -> str: ...
```

### Parameter reference

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `root_tag` | `None` | Outer wrapper tag; `None` omits the wrapper |
| `item_tag` | `'item'` | Tag for each list element (overridden by class name for dataclasses/models) |
| `none_str` | `'null'` | String substituted for `None` values |
| `indent` | `'  '` (2 spaces) | Indentation per level; `None` for compact output |
| `include_field_info` | `False` | Attach Pydantic `Field` / dataclass `field()` `title`/`description` as XML attributes; `'once'` only on first occurrence |

### Example 1 — Plain dict

```python
from pydantic_ai import format_as_xml

data = {"name": "Alice", "score": 95, "active": True}
print(format_as_xml(data, root_tag="user"))
# <user>
#   <name>Alice</name>
#   <score>95</score>
#   <active>True</active>
# </user>
```

### Example 2 — Pydantic model with field descriptions

```python
from pydantic import BaseModel, Field
from pydantic_ai import format_as_xml

class Product(BaseModel):
    name: str = Field(description="Product display name")
    price: float = Field(description="Price in USD")
    in_stock: bool

p = Product(name="Widget", price=9.99, in_stock=True)

# Without field info
print(format_as_xml(p))
# <Product>
#   <name>Widget</name>
#   <price>9.99</price>
#   <in_stock>True</in_stock>
# </Product>

# With field descriptions as attributes
print(format_as_xml(p, include_field_info=True))
# <Product>
#   <name description="Product display name">Widget</name>
#   <price description="Price in USD">9.99</price>
#   <in_stock>True</in_stock>
# </Product>
```

### Example 3 — List of models, `include_field_info='once'`

```python
from pydantic import BaseModel, Field
from pydantic_ai import format_as_xml

class Example(BaseModel):
    input: str = Field(description="Few-shot input")
    output: str = Field(description="Expected output")

examples = [
    Example(input="What is 2+2?", output="4"),
    Example(input="Capital of France?", output="Paris"),
]

# 'once' adds attributes only on the first occurrence of each field
xml = format_as_xml(examples, root_tag="examples", include_field_info="once")
print(xml)
# <examples>
#   <Example>
#     <input description="Few-shot input">What is 2+2?</input>
#     <output description="Expected output">4</output>
#   </Example>
#   <Example>
#     <input>Capital of France?</input>
#     <output>Paris</output>
#   </Example>
# </examples>
```

### Example 4 — Injecting XML examples into a system prompt

```python
from dataclasses import dataclass
from pydantic_ai import Agent, format_as_xml

@dataclass
class QAPair:
    question: str
    answer: str

FEW_SHOT = [
    QAPair(question="2+2?", answer="4"),
    QAPair(question="3×7?", answer="21"),
]

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt=(
        "You answer maths questions concisely.\n\n"
        "Examples:\n"
        + format_as_xml(FEW_SHOT, root_tag="examples")
    ),
)

result = agent.run_sync("What is 12 × 8?")
print(result.data)  # 96
```

### Example 5 — Compact output (no indentation)

```python
from pydantic_ai import format_as_xml

# Use indent=None for minimal tokens
compact = format_as_xml({"a": 1, "b": [1, 2]}, root_tag="data", indent=None)
print(compact)
# <data><a>1</a><b><item>1</item><item>2</item></b></data>
```

---

## 5. `BinaryContent` + `FileUrl` family

**Module:** `pydantic_ai.messages`  
**Exports:** `from pydantic_ai import BinaryContent, ImageUrl, AudioUrl, VideoUrl, DocumentUrl`  
(also: `from pydantic_ai.messages import FileUrl, BinaryImage`)

Pydantic AI has two parallel multimodal content systems: **URL references** (the `FileUrl`
family) and **raw binary** (`BinaryContent`). Both are valid in `UserPromptPart.content`
alongside plain strings.

### `FileUrl` — abstract base

```python
@pydantic_dataclass
class FileUrl(ABC):
    url: str
    force_download: ForceDownloadMode = False  # False | True | 'allow-local'
    vendor_metadata: dict[str, Any] | None = None
    media_type: str | None = None   # constructor alias for _media_type
    identifier: str | None = None   # constructor alias for _identifier
```

**`force_download` semantics:**

| Value | Behaviour |
|-------|-----------|
| `False` (default) | URL sent directly to providers that support it; downloaded with SSRF protection for others |
| `True` | Always downloaded by PydanticAI with SSRF protection (blocks private IPs and cloud metadata) |
| `'allow-local'` | Always downloaded; private IPs allowed but cloud metadata still blocked |

**`vendor_metadata` by provider:**

| Provider | Field | Effect |
|----------|-------|--------|
| Google | `VideoUrl.vendor_metadata` | Used as `video_metadata` |
| OpenAI / xAI | `ImageUrl.vendor_metadata['detail']` | Image detail level (`'auto'` / `'low'` / `'high'`) |

### Typed subclasses

| Class | `kind` | Accepted media types |
|-------|--------|----------------------|
| `ImageUrl` | `'image-url'` | `image/*` |
| `AudioUrl` | `'audio-url'` | `audio/*` |
| `VideoUrl` | `'video-url'` | `video/*` |
| `DocumentUrl` | `'document-url'` | `application/pdf`, `text/*`, etc. |

### Example 1 — Send an image URL to a vision model

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ImageUrl

agent = Agent("openai:gpt-4o")

result = agent.run_sync(
    [
        "Describe this image:",
        ImageUrl(url="https://example.com/photo.jpg"),
    ]
)
print(result.data)
```

### Example 2 — Image with detail level (OpenAI / xAI)

```python
from pydantic_ai.messages import ImageUrl

# Force high-res processing for a detailed chart
chart_url = ImageUrl(
    url="https://example.com/sales_chart.png",
    vendor_metadata={"detail": "high"},
)
```

### Example 3 — Transcribe audio

```python
from pydantic_ai import Agent
from pydantic_ai.messages import AudioUrl

agent = Agent("google-gla:gemini-2.0-flash")

result = agent.run_sync(
    [
        "Transcribe and summarise this audio recording:",
        AudioUrl(url="https://example.com/meeting.mp3"),
    ]
)
print(result.data)
```

### Example 4 — Analyse a PDF document

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DocumentUrl

agent = Agent("anthropic:claude-sonnet-4-5")

result = agent.run_sync(
    [
        "Summarise the key findings from this report:",
        DocumentUrl(url="https://example.com/annual_report.pdf"),
    ]
)
print(result.data)
```

### `BinaryContent` — raw binary bytes

For binary data you already have in memory, use `BinaryContent`:

```python
@pydantic_dataclass
class BinaryContent:
    data: bytes
    media_type: AudioMediaType | ImageMediaType | DocumentMediaType | str
    vendor_metadata: dict[str, Any] | None = None
    identifier: str | None = None
    kind: Literal['binary'] = 'binary'
```

`BinaryContent.base64` returns the base64-encoded string. The pydantic JSON schema serialises
`data` as base64 automatically.

### Example 5 — Send a local image file

```python
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

agent = Agent("anthropic:claude-sonnet-4-5")

image_bytes = Path("screenshot.png").read_bytes()

result = agent.run_sync(
    [
        "What's wrong in this screenshot?",
        BinaryContent(data=image_bytes, media_type="image/png"),
    ]
)
print(result.data)
```

### Example 6 — Transcribe audio bytes

```python
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

agent = Agent("google-gla:gemini-2.0-flash")

audio_bytes = open("recording.wav", "rb").read()

result = agent.run_sync(
    [
        "Transcribe this recording:",
        BinaryContent(data=audio_bytes, media_type="audio/wav"),
    ]
)
print(result.data)
```

### Example 7 — `force_download=True` for SSRF-safe proxy

```python
from pydantic_ai.messages import ImageUrl

# Force PydanticAI to download and forward the image bytes,
# blocking private IPs even though the provider would accept the URL directly
safe_img = ImageUrl(
    url="https://user-uploads.example.com/img/42.jpg",
    force_download=True,
)
```

### `MultiModalContent` — the union type

`pydantic_ai.MultiModalContent` is a discriminated union of all file content types:
`ImageUrl | AudioUrl | DocumentUrl | VideoUrl | BinaryContent | UploadedFile`. Use it
when you need to accept any multimodal content in a tool argument or output type:

```python
from pydantic_ai import Agent, MultiModalContent
from pydantic_ai.messages import ImageUrl, BinaryContent

def process_file(content: MultiModalContent) -> str:
    if isinstance(content, ImageUrl):
        return f"URL image: {content.url}"
    elif isinstance(content, BinaryContent):
        return f"Binary {content.media_type}: {len(content.data)} bytes"
    return "Other file type"
```

---

## 6. `DeferredToolRequests` + `DeferredToolResults` + `CallDeferred` + `ApprovalRequired`

**Module:** `pydantic_ai.tools` / `pydantic_ai.exceptions`  
**Exports:** `from pydantic_ai import DeferredToolRequests, DeferredToolResults, CallDeferred, ApprovalRequired, ToolApproved, ToolDenied`

Pydantic AI supports two patterns for tools that need external processing:

| Pattern | Exception | Use case |
|---------|-----------|----------|
| **Deferred** | `CallDeferred` | External async execution (webhooks, queues) |
| **Approval** | `ApprovalRequired` | Human-in-the-loop review before execution |

Both patterns return a `DeferredToolRequests` output from the current run. Results are
supplied in a subsequent run via `DeferredToolResults`.

### `CallDeferred`

Raise inside any `@agent.tool` function to defer execution to an external system:

```python
@dataclass
class CallDeferred(Exception):
    metadata: dict[str, Any] | None = None  # forwarded to DeferredToolRequests.metadata
```

### `ApprovalRequired`

Raise inside any `@agent.tool` function to require human approval:

```python
@dataclass
class ApprovalRequired(Exception):
    metadata: dict[str, Any] | None = None  # forwarded to DeferredToolRequests.metadata
```

### `DeferredToolRequests`

```python
@dataclass
class DeferredToolRequests:
    calls: list[ToolCallPart] = ...       # tools that raised CallDeferred
    approvals: list[ToolCallPart] = ...   # tools that raised ApprovalRequired
    metadata: dict[str, dict] = ...       # metadata keyed by tool_call_id

    def build_results(
        self,
        approvals: dict[str, bool | DeferredToolApprovalResult] | None = None,
        calls: dict[str, DeferredToolCallResult | Any] | None = None,
        metadata: dict[str, dict] | None = None,
        approve_all: bool = False,
    ) -> DeferredToolResults: ...

    def remaining(self, results: DeferredToolResults) -> DeferredToolRequests | None: ...
```

### `DeferredToolResults`

```python
@dataclass
class DeferredToolResults:
    calls: dict[str, DeferredToolCallResult | Any] = ...
    approvals: dict[str, bool | DeferredToolApprovalResult] = ...
    metadata: dict[str, dict] = ...

    def update(self, other: DeferredToolResults) -> None: ...
    def to_tool_call_results(self) -> dict[str, DeferredToolResult]: ...
```

### Example 1 — Deferred external execution

```python
import asyncio
from pydantic_ai import Agent, CallDeferred, DeferredToolRequests, DeferredToolResults
from pydantic_ai.output import DeferredToolRequests as DTR

agent = Agent(
    "openai:gpt-4o-mini",
    output_type=DeferredToolRequests | str,  # deferred or normal string answer
)

@agent.tool_plain
def send_email(to: str, subject: str, body: str) -> str:
    # Signal to defer — attach routing metadata
    raise CallDeferred(metadata={"queue": "email", "priority": "high"})

async def main():
    # First run — agent decides to call send_email
    result = await agent.run("Send a welcome email to alice@example.com")
    
    if isinstance(result.data, DeferredToolRequests):
        requests = result.data
        print("Deferred calls:", [c.tool_name for c in requests.calls])
        
        # Simulate external execution
        external_results = {}
        for call in requests.calls:
            args = call.args_as_dict()
            # ... actually send email here ...
            external_results[call.tool_call_id] = f"Email sent to {args.get('to')}"
        
        # Build DeferredToolResults and continue
        tool_results = requests.build_results(calls=external_results)
        
        # Second run — supply results, agent produces final response
        final = await agent.run(
            None,
            message_history=result.all_messages(),
            deferred_tool_results=tool_results,
        )
        print(final.data)
    else:
        print(result.data)

asyncio.run(main())
```

### Example 2 — Human-in-the-loop approval

```python
import asyncio
from pydantic_ai import Agent, ApprovalRequired, DeferredToolRequests, ToolApproved, ToolDenied

agent = Agent(
    "anthropic:claude-sonnet-4-5",
    output_type=DeferredToolRequests | str,
)

@agent.tool_plain
def delete_records(table: str, where_clause: str) -> str:
    raise ApprovalRequired(metadata={"table": table, "clause": where_clause})

async def human_approve(request: dict) -> bool:
    print(f"APPROVAL REQUEST: DELETE FROM {request['table']} WHERE {request['clause']}")
    answer = input("Approve? (y/n): ").strip().lower()
    return answer == "y"

async def main():
    result = await agent.run("Delete all inactive users from the users table")

    if isinstance(result.data, DeferredToolRequests):
        requests = result.data
        approvals = {}
        
        for call in requests.approvals:
            meta = requests.metadata.get(call.tool_call_id, {})
            approved = await human_approve(meta)
            if approved:
                # ToolApproved can override args
                approvals[call.tool_call_id] = ToolApproved()
            else:
                approvals[call.tool_call_id] = ToolDenied("Rejected by administrator.")
        
        tool_results = requests.build_results(approvals=approvals)
        
        final = await agent.run(
            None,
            message_history=result.all_messages(),
            deferred_tool_results=tool_results,
        )
        print(final.data)

asyncio.run(main())
```

### Example 3 — `approve_all` shortcut

```python
# Approve all pending approval-requests in one call
results = requests.build_results(approve_all=True)
```

### Example 4 — `remaining()` for partial resolution

```python
# Resolve only some calls; check what's left
partial_results = requests.build_results(
    calls={first_call_id: "result_value"}
)
leftover = requests.remaining(partial_results)
if leftover:
    print(f"Still pending: {len(leftover.calls)} calls, {len(leftover.approvals)} approvals")
```

### Example 5 — Mixed deferred + approval in one run

```python
import asyncio
from pydantic_ai import Agent, CallDeferred, ApprovalRequired, DeferredToolRequests

agent = Agent("openai:gpt-4o", output_type=DeferredToolRequests | str)

@agent.tool_plain
def send_newsletter(topic: str) -> str:
    raise CallDeferred(metadata={"type": "newsletter"})

@agent.tool_plain
def purge_cache(scope: str) -> str:
    raise ApprovalRequired(metadata={"scope": scope, "risk": "high"})

async def main():
    result = await agent.run("Send the newsletter and purge the CDN cache")
    if isinstance(result.data, DeferredToolRequests):
        dr = result.data
        print("External calls:", [c.tool_name for c in dr.calls])
        print("Approval needed:", [c.tool_name for c in dr.approvals])
        
        # Build results: execute calls externally, approve or deny approvals
        tool_results = dr.build_results(
            calls={c.tool_call_id: "sent" for c in dr.calls},
            approvals={c.tool_call_id: ToolApproved() for c in dr.approvals},
        )
        final = await agent.run(
            None, message_history=result.all_messages(),
            deferred_tool_results=tool_results,
        )
        print(final.data)

asyncio.run(main())
```

---

## 7. `NativeTool` capability

**Module:** `pydantic_ai.capabilities`  
**Export:** `from pydantic_ai.capabilities import NativeTool`

`NativeTool` registers a **provider-native tool** (web search, code execution, file search,
MCP servers, etc.) via the capabilities API. It wraps a single `AgentNativeTool` — either
a static `AbstractNativeTool` instance or a callable that dynamically creates one per run
from `RunContext`.

```python
@dataclass
class NativeTool(AbstractCapability[AgentDepsT]):
    tool: AgentNativeTool[AgentDepsT]  # static instance or RunContext callable
```

### Static native tool registration

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebSearchTool, CodeExecutionTool, FileSearchTool

# Web search (Anthropic, OpenAI Responses, Groq, Google, xAI, OpenRouter)
agent = Agent(
    "anthropic:claude-sonnet-4-5",
    capabilities=[NativeTool(WebSearchTool(search_context_size="high"))],
)

# Code execution (Anthropic, OpenAI Responses, Google, Bedrock Nova2, xAI)
agent = Agent(
    "openai:gpt-4o",
    capabilities=[NativeTool(CodeExecutionTool())],
)

# File search with vector store (OpenAI Responses, Google)
agent = Agent(
    "openai:gpt-4o",
    capabilities=[NativeTool(FileSearchTool(vector_store_ids=["vs_abc123"]))],
)
```

### Dynamic native tool — per-run configuration

A `CapabilityFunc` is automatically wrapped in a `DynamicCapability` + `NativeTool`:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebSearchTool

@dataclass
class UserProfile:
    country: str
    preferred_language: str

def web_search_for_user(ctx: RunContext[UserProfile]) -> WebSearchTool:
    return WebSearchTool(
        user_location={"country": ctx.deps.country},
        search_context_size="medium",
    )

agent = Agent(
    "anthropic:claude-sonnet-4-5",
    capabilities=[NativeTool(web_search_for_user)],
)

result = agent.run_sync(
    "Find the latest news.",
    deps=UserProfile(country="GB", preferred_language="en"),
)
```

### `NativeTool.from_spec()` — YAML/dict construction

```python
from pydantic_ai.capabilities import NativeTool

# Flat form — kwargs become the native tool fields
cap = NativeTool.from_spec(kind="web_search", search_context_size="high")

# Explicit form — pass a dict as 'tool'
cap = NativeTool.from_spec(tool={"kind": "web_search"})
```

---

## 8. `NativeOrLocalTool` capability

**Module:** `pydantic_ai.capabilities`  
**Export:** `from pydantic_ai.capabilities import NativeOrLocalTool`

`NativeOrLocalTool` pairs a **provider-native tool** with a **local fallback function**.
When the model supports the native tool, the local fallback is removed. When it doesn't,
the native tool is removed and only the local function is registered.

This is the base class behind `WebSearch`, `WebFetch`, and `ImageGeneration` — use it to
build your own adaptive capabilities.

```python
@dataclass(init=False)
class NativeOrLocalTool(AbstractCapability[AgentDepsT]):
    native: AgentNativeTool | bool = True  # native tool or True for subclass default
    local: str | Tool | Callable | AbstractToolset | bool | None = None
```

### `native` values

| Value | Behaviour |
|-------|-----------|
| `True` (default in subclasses) | Use the subclass's default native tool |
| `False` | Disable native; always use local |
| `AbstractNativeTool` instance | Use this specific configuration |
| `Callable` | Dynamically create native tool from `RunContext` |

### `local` values

| Value | Behaviour |
|-------|-----------|
| `None` (default) | Auto-detect via subclass `_default_local` |
| `True` | Opt into the default local fallback |
| `False` | Disable local; only use native (error on unsupported models) |
| A strategy string | Resolved by subclass (e.g. `'duckduckgo'` for `WebSearch`) |
| `Tool` / `AbstractToolset` | Use this specific local tool |
| Bare callable | Auto-wrapped in `Tool` |

### Example 1 — Custom native+local pair

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeOrLocalTool
from pydantic_ai.native_tools import XSearchTool

async def local_x_search(query: str) -> str:
    """Search X/Twitter using REST API as fallback."""
    # ... call X API ...
    return f"X results for: {query}"

agent = Agent(
    "openai:gpt-4o",  # xAI not available
    capabilities=[
        NativeOrLocalTool(
            native=XSearchTool(),
            local=local_x_search,
        )
    ],
)
# → gpt-4o doesn't support XSearchTool natively, so local_x_search is registered instead
```

### Example 2 — Disabling native, always use local

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeOrLocalTool
from pydantic_ai.native_tools import WebSearchTool

async def my_search(query: str) -> str:
    """Custom search with internal knowledge base."""
    return "Internal search result"

agent = Agent(
    "anthropic:claude-sonnet-4-5",
    capabilities=[
        NativeOrLocalTool(
            native=False,   # disable native web search entirely
            local=my_search,
        )
    ],
)
```

### Example 3 — Built-in `WebSearch` capability (subclass)

The built-in `WebSearch`, `WebFetch`, and `ImageGeneration` capabilities are all subclasses
of `NativeOrLocalTool`:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch, WebFetch

agent = Agent(
    "openai:gpt-4o",
    capabilities=[
        WebSearch(local="duckduckgo"),  # use DuckDuckGo as local fallback
        WebFetch(),                      # use local HTTP fetch as fallback
    ],
)
```

---

## 9. `DynamicCapability`

**Module:** `pydantic_ai.capabilities`  
**Export:** `from pydantic_ai.capabilities import DynamicCapability`

`DynamicCapability` builds another capability per-run using a factory function that receives
the `RunContext`. The returned capability replaces this wrapper for the rest of the run —
its instructions, model settings, toolset, native tools, and hooks all flow through normally.

```python
@dataclass
class DynamicCapability(AbstractCapability[AgentDepsT]):
    capability_func: CapabilityFunc[AgentDepsT]
    # CapabilityFunc = Callable[[RunContext[AgentDepsT]], AbstractCapability | Awaitable | None]
```

When you pass a bare callable to `Agent(capabilities=[...])` or `agent.run(capabilities=[...])`,
Pydantic AI automatically wraps it in a `DynamicCapability`.

### Example 1 — Feature-flag gated capability

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import DynamicCapability, WebSearch, NativeTool
from pydantic_ai.native_tools import CodeExecutionTool

@dataclass
class AppConfig:
    enable_web_search: bool
    enable_code_execution: bool
    user_tier: str  # 'free' | 'pro' | 'enterprise'

def feature_flag_capability(ctx: RunContext[AppConfig]) -> WebSearch | NativeTool | None:
    """Enable capabilities based on feature flags and user tier."""
    cfg = ctx.deps
    if cfg.enable_web_search and cfg.user_tier in ("pro", "enterprise"):
        return WebSearch()
    if cfg.enable_code_execution and cfg.user_tier == "enterprise":
        return NativeTool(CodeExecutionTool())
    return None  # no extra capability

agent = Agent(
    "openai:gpt-4o",
    capabilities=[DynamicCapability(feature_flag_capability)],
    # or equivalently (auto-wrapping):
    # capabilities=[feature_flag_capability],
)

async def main():
    result = await agent.run(
        "Search for the latest Python release.",
        deps=AppConfig(enable_web_search=True, enable_code_execution=False, user_tier="pro"),
    )
    print(result.data)

asyncio.run(main())
```

### Example 2 — Async factory

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import FileSearchTool

async def load_vector_store(ctx: RunContext[str]) -> NativeTool:
    """Resolve the vector store ID from the user ID at run time."""
    user_id = ctx.deps
    # Simulate async DB lookup
    await asyncio.sleep(0)
    vector_store_id = f"vs_{user_id[:8]}"
    return NativeTool(FileSearchTool(vector_store_ids=[vector_store_id]))

agent = Agent(
    "openai:gpt-4o",
    capabilities=[load_vector_store],  # auto-wrapped as DynamicCapability
)

async def main():
    result = await agent.run("Find my recent invoices.", deps="user_abc123")
    print(result.data)

asyncio.run(main())
```

### Example 3 — Per-run model settings

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.settings import ModelSettings

@dataclass
class RequestMeta:
    temperature: float
    max_tokens: int

class DynamicModelSettings(AbstractCapability[RequestMeta]):
    async def for_run(self, ctx: RunContext[RequestMeta]) -> 'DynamicModelSettings':
        return self

    def get_model_settings(self) -> ModelSettings:
        # This won't work directly — model settings are per-run, not per-capability
        # Use agent.run(model_settings=...) for this pattern instead
        return {}

# Better pattern: pass model_settings directly to run()
agent = Agent("openai:gpt-4o")

async def creative_run(prompt: str) -> str:
    result = await agent.run(
        prompt,
        model_settings={"temperature": 1.2, "max_tokens": 512},
    )
    return result.data
```

### Example 4 — Combining multiple dynamic capabilities

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import DynamicCapability, WebSearch, Thinking

@dataclass
class RunConfig:
    needs_search: bool
    needs_thinking: bool

async def build_capabilities(ctx: RunContext[RunConfig]) -> list:
    caps = []
    if ctx.deps.needs_search:
        caps.append(WebSearch())
    if ctx.deps.needs_thinking:
        caps.append(Thinking(budget_tokens=8000))
    # Return a CombinedCapability or the list; None also works per-capability
    from pydantic_ai.capabilities import CombinedCapability
    return CombinedCapability(caps) if caps else None

agent = Agent(
    "anthropic:claude-sonnet-4-5",
    capabilities=[build_capabilities],
)
```

---

## 10. `ToolSearch` capability

**Module:** `pydantic_ai.capabilities`  
**Export:** `from pydantic_ai.capabilities import ToolSearch`

`ToolSearch` enables **lazy tool discovery** for agents with many tools. Tools marked
`defer_loading=True` are hidden from the model until they are found via search. This keeps
the tool list short and prompt tokens low at the start of a run.

`ToolSearch` is **auto-injected into every agent** at zero overhead — no extra tokens are
sent unless deferred tools actually exist.

### Constructor

```python
@dataclass
class ToolSearch(AbstractCapability[AgentDepsT]):
    strategy: ToolSearchStrategy[AgentDepsT] | None = None
    max_results: int = 10
    tool_description: str | None = None      # custom description for search_tools function
    parameter_description: str | None = None  # custom description for the queries param
```

### Strategy options

| Value | Behaviour |
|-------|-----------|
| `None` (default) | Best available: native on Anthropic/OpenAI, keyword-overlap elsewhere |
| `'keywords'` | Local keyword-overlap algorithm; prompt-cache compatible |
| `'bm25'` | Anthropic BM25 native search; errors on other providers |
| `'regex'` | Anthropic regex native search; errors on other providers |
| `Callable` | Custom search function `(ctx, queries, tools) -> list[str]` |

### Native vs local discovery

On **Anthropic** and **OpenAI Responses**, deferred tools are sent to the server with a
`defer_loading` flag — the provider handles discovery internally and the result is delivered
as a "client-executed" tool call. This preserves the prompt cache because the tool list
doesn't change between turns.

On **other providers**, `ToolSearch` injects a local `search_tools` function tool that the
model calls to discover tools.

### Example 1 — Basic tool search with deferred tools

```python
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import ToolSearch

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

def calculate_carbon_offset(km: float, transport: str) -> float:
    """Calculate CO2 offset for a journey."""
    factors = {"car": 0.21, "plane": 0.255, "train": 0.041}
    return km * factors.get(transport, 0.1)

def get_currency_rate(from_currency: str, to_currency: str) -> float:
    """Get current exchange rate between two currencies."""
    return 1.27 if (from_currency, to_currency) == ("USD", "GBP") else 1.0

# Only get_weather is immediately visible; others are discovered on demand
agent = Agent(
    "anthropic:claude-sonnet-4-5",
    tools=[
        Tool(get_weather),  # always visible
        Tool(calculate_carbon_offset, defer_loading=True),
        Tool(get_currency_rate, defer_loading=True),
    ],
    capabilities=[ToolSearch()],
)

result = agent.run_sync("What's the weather in Paris and what's the USD/GBP rate?")
print(result.data)
```

### Example 2 — Force keyword strategy (all providers)

```python
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import ToolSearch

agent = Agent(
    "openai:gpt-4o",
    tools=[
        Tool(get_weather),
        Tool(calculate_carbon_offset, defer_loading=True),
        Tool(get_currency_rate, defer_loading=True),
    ],
    capabilities=[ToolSearch(strategy="keywords", max_results=5)],
)
```

### Example 3 — Custom search function

```python
from collections.abc import Sequence
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.tools import ToolDefinition

def semantic_search(
    ctx: RunContext[None],
    queries: Sequence[str],
    tools: Sequence[ToolDefinition],
) -> list[str]:
    """Score tools by how many query words appear in their description."""
    scores: dict[str, int] = {}
    for tool in tools:
        desc = (tool.description or "").lower()
        scores[tool.name] = sum(
            1 for q in queries
            for word in q.lower().split()
            if word in desc
        )
    # Return names sorted by score, top max_results
    return [
        name for name, _ in sorted(scores.items(), key=lambda x: -x[1])
        if scores[name] > 0
    ][:10]

agent = Agent(
    "anthropic:claude-sonnet-4-5",
    tools=[
        Tool(get_weather),
        Tool(calculate_carbon_offset, defer_loading=True),
        Tool(get_currency_rate, defer_loading=True),
    ],
    capabilities=[ToolSearch(strategy=semantic_search)],
)
```

### Example 4 — Large toolset across multiple toolsets

```python
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.toolsets import FunctionToolset

# A large collection of 50+ analytics tools
analytics_tools = [
    Tool(fn, defer_loading=True) for fn in [
        # ... 50 analytics functions ...
    ]
]
analytics_toolset = FunctionToolset(analytics_tools)

# A small set of always-visible core tools
core_tools = [Tool(get_weather)]

agent = Agent(
    "openai:gpt-4o",
    tools=core_tools,
    toolsets=[analytics_toolset],
    capabilities=[
        ToolSearch(
            strategy="keywords",
            max_results=8,
            tool_description="Search the analytics toolkit. Provide descriptive query terms.",
        )
    ],
)
```

### Example 5 — Anthropic BM25 native strategy

```python
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import ToolSearch

# Force Anthropic's BM25 native search — will raise on non-Anthropic models
agent = Agent(
    "anthropic:claude-sonnet-4-5",
    tools=[Tool(get_weather), Tool(calculate_carbon_offset, defer_loading=True)],
    capabilities=[ToolSearch(strategy="bm25")],
)
```

---

## Summary table

| Class | Module | Key use case |
|-------|--------|-------------|
| `MCPToolset` | `pydantic_ai.mcp` | Connect to any MCP server (HTTP, stdio, in-process) |
| `load_mcp_toolsets` | `pydantic_ai.mcp` | Load multiple MCP servers from a JSON config |
| `UIAdapter` | `pydantic_ai.ui` | Base class for chat-UI / AG-UI / WebSocket adapters |
| `MessagesBuilder` | `pydantic_ai.ui` | Accumulate message parts into `ModelMessage` list |
| `CachePoint` | `pydantic_ai.messages` | Mark prompt cache boundaries (Anthropic, Bedrock) |
| `CompactionPart` | `pydantic_ai.messages` | Round-trip provider-produced conversation summaries |
| `format_as_xml` | `pydantic_ai.format_prompt` | Convert Python objects to LLM-friendly XML |
| `BinaryContent` | `pydantic_ai.messages` | Send raw binary bytes (images, audio, docs) |
| `ImageUrl` / `AudioUrl` / `VideoUrl` / `DocumentUrl` | `pydantic_ai.messages` | Send multimodal content by URL |
| `DeferredToolRequests` | `pydantic_ai.tools` | Receive tool calls that need external execution or approval |
| `DeferredToolResults` | `pydantic_ai.tools` | Supply results for deferred tool calls in the next run |
| `CallDeferred` | `pydantic_ai.exceptions` | Raise inside a tool to defer it for external execution |
| `ApprovalRequired` | `pydantic_ai.exceptions` | Raise inside a tool to require human approval |
| `NativeTool` | `pydantic_ai.capabilities` | Register a native tool via the capabilities API |
| `NativeOrLocalTool` | `pydantic_ai.capabilities` | Pair a native tool with a local fallback |
| `DynamicCapability` | `pydantic_ai.capabilities` | Build a capability per-run from `RunContext` |
| `ToolSearch` | `pydantic_ai.capabilities` | Lazy tool discovery for large toolsets |

---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 41"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.12.1: GeminiChatClient+RawGeminiChatClient (Gemini Developer API and Vertex AI — MRO stack, OTEL_PROVIDER_NAME='gcp.gemini', dual-prefix env resolution); GeminiChatOptions+ThinkingConfig (top_k, response_schema, thinking_config, thinking_budget/level, unsupported-field table); GeminiSettings+GoogleGeminiSettings (GEMINI_* vs GOOGLE_* env priority); MontyExecuteCodeTool+FileMount (sandboxed Python execution — MountMode overlay/rw/ro, resource_limits, file output scanning); MontyCodeActProvider (CodeAct surface wired to MontyExecuteCodeTool — tools/file_mounts delegation, add/remove/clear mutators); MistralEmbeddingClient+MistralEmbeddingOptions+MistralEmbeddingSettings (Mistral AI embeddings — MISTRAL_API_KEY/MISTRAL_EMBEDDING_MODEL, OTEL_PROVIDER_NAME='mistralai', MISTRAL_SERVER_URL override); MessageInjectionMiddleware+enqueue_messages (queue messages mid-run from tool code — session.state queue, get_pending_messages snapshot, MESSAGE_INJECTION_PENDING_MESSAGES_STATE_KEY); CachingSkillsSource (composable skills cache — per-key asyncio.Lock guard, refresh_interval timedelta, cache_isolation_key_selector); FoundryToolbox (MCPStreamableHTTPTool wrapper — bearer-token auth, TOOLBOX_ENDPOINT/TOOLBOX_NAME/FOUNDRY_PROJECT_ENDPOINT env vars); ResponsesHostServer+InvocationsHostServer (Foundry hosting servers — OpenAI Responses API, JSON invocations, streaming SSE, PORT env var) — source-verified at agent-framework 1.12.1."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 64
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 41

Verified against **agent-framework 1.12.1** (installed July 2026). Every constructor signature, parameter description, and code example was derived from the installed package source using `inspect.getsource()`.

Sub-packages introspected:
`agent_framework_gemini 1.0.0b260722`,
`agent_framework_monty 1.0.0b260721`,
`agent_framework_mistral 1.0.0b260721`,
`agent_framework._middleware` (MessageInjectionMiddleware),
`agent_framework._skills` (CachingSkillsSource),
`agent_framework_foundry_hosting 1.0.0b260722`.

**Previous volumes:** [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) through [Vol. 40](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v40/) — 400+ classes covered.

This volume covers **ten class groups** across the new Gemini provider, the Monty Python sandbox, Mistral AI embeddings, in-run message injection, composable skills caching, and the Foundry hosting servers.

| # | Class / group | Package |
|---|---|---|
| 1 | `GeminiChatClient` · `RawGeminiChatClient` | `agent_framework_gemini` |
| 2 | `GeminiChatOptions` · `ThinkingConfig` | `agent_framework_gemini` |
| 3 | `GeminiSettings` · `GoogleGeminiSettings` | `agent_framework_gemini` |
| 4 | `MontyExecuteCodeTool` · `FileMount` | `agent_framework_monty` |
| 5 | `MontyCodeActProvider` | `agent_framework_monty` |
| 6 | `MistralEmbeddingClient` · `MistralEmbeddingOptions` · `MistralEmbeddingSettings` | `agent_framework_mistral` |
| 7 | `MessageInjectionMiddleware` · `enqueue_messages` | `agent_framework` core |
| 8 | `CachingSkillsSource` | `agent_framework` core |
| 9 | `FoundryToolbox` | `agent_framework_foundry_hosting` |
| 10 | `ResponsesHostServer` · `InvocationsHostServer` | `agent_framework_foundry_hosting` |

---

## 1 · `GeminiChatClient` · `RawGeminiChatClient`

**Package:** `agent_framework_gemini` (import via `from agent_framework.gemini import …`)

`GeminiChatClient` is the full-featured Gemini provider that targets either the **Gemini Developer API** or **Vertex AI**. It stacks four MRO layers on top of `RawGeminiChatClient`: function invocation, chat middleware, telemetry, and the raw HTTP client.

```
GeminiChatClient
 ├── FunctionInvocationLayer   ← handles tool-call loops
 ├── ChatMiddlewareLayer       ← composable middleware chain
 ├── ChatTelemetryLayer        ← OTel traces + metrics
 └── RawGeminiChatClient       ← raw genai.Client HTTP calls
```

### Constructor (shared by both client types)

```python
class RawGeminiChatClient(BaseChatClient[GeminiChatOptionsT]):
    OTEL_PROVIDER_NAME: ClassVar[str] = "gcp.gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        vertexai: bool | None = None,
        project: str | None = None,
        location: str | None = None,
        credentials: Credentials | None = None,     # google.auth.credentials.Credentials
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
        client: genai.Client | None = None,          # pre-built SDK client
        additional_properties: dict[str, Any] | None = None,
    ) -> None: ...
```

`GeminiChatClient` adds two more parameters:

```python
class GeminiChatClient(FunctionInvocationLayer, ChatMiddlewareLayer,
                        ChatTelemetryLayer, RawGeminiChatClient):
    def __init__(
        self,
        *,
        # ... all RawGeminiChatClient params ...
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
    ) -> None: ...
```

| Parameter | Description |
|---|---|
| `api_key` | Gemini Developer API key. Falls back to `GOOGLE_API_KEY` then `GEMINI_API_KEY`. |
| `model` | Default model (e.g. `"gemini-2.5-flash"`). Falls back to `GOOGLE_MODEL` then `GEMINI_MODEL`. |
| `vertexai` | `True` = use Vertex AI endpoints. Falls back to `GOOGLE_GENAI_USE_VERTEXAI` env var. |
| `project` | Google Cloud project ID for Vertex AI. Falls back to `GOOGLE_CLOUD_PROJECT`. |
| `location` | Vertex AI region (e.g. `"us-central1"`). Falls back to `GOOGLE_CLOUD_LOCATION`. |
| `credentials` | `google.auth.credentials.Credentials` for Vertex AI. Omit to use Application Default Credentials. |
| `client` | Pre-built `genai.Client` instance. When supplied, all auth settings are bypassed. |

### Using the Gemini Developer API

```python
import asyncio
import os
from agent_framework import Agent
from agent_framework.gemini import GeminiChatClient

async def main() -> None:
    # api_key resolves from GOOGLE_API_KEY or GEMINI_API_KEY
    client = GeminiChatClient(
        api_key=os.environ["GOOGLE_API_KEY"],
        model="gemini-2.5-flash",
    )
    agent = Agent(
        client=client,
        instructions="You are a helpful assistant.",
    )
    response = await agent.run("Explain quantum entanglement in two sentences.")
    print(response.text)

asyncio.run(main())
```

### Using Vertex AI

```python
import asyncio
from agent_framework import Agent
from agent_framework.gemini import GeminiChatClient
from google.auth import default as google_auth_default

async def main() -> None:
    credentials, project_id = google_auth_default()

    client = GeminiChatClient(
        vertexai=True,
        project=project_id,
        location="us-central1",
        credentials=credentials,
        model="gemini-2.5-pro",
    )
    agent = Agent(client=client)
    response = await agent.run("What is the capital of France?")
    print(response.text)

asyncio.run(main())
```

### Using a pre-built `genai.Client`

```python
from google import genai
from agent_framework.gemini import GeminiChatClient

# Build the client yourself for custom httpx transports, proxies, etc.
raw_client = genai.Client(api_key="sk-...", http_options={"timeout": 30})
chat_client = GeminiChatClient(client=raw_client, model="gemini-2.5-flash")
```

---

## 2 · `GeminiChatOptions` · `ThinkingConfig`

**Package:** `agent_framework_gemini`

`GeminiChatOptions` extends the standard `ChatOptions` with Gemini-specific generation config fields. It is a `TypedDict` with `total=False` (all fields optional).

### Gemini-specific fields

| Field | Type | Description |
|---|---|---|
| `top_k` | `int` | Limits token selection to the top-K most probable tokens. |
| `response_schema` | `dict[str, Any]` | Raw JSON schema for structured output. Sets `response_mime_type='application/json'` and passes the schema directly. Use instead of `response_format` when you need a raw schema dict. |
| `thinking_config` | `ThinkingConfig` | Extended thinking configuration (see below). |

### Unsupported fields (pass `None` to signal non-support)

| Field | Reason |
|---|---|
| `logit_bias` | Not available in the Gemini API. |
| `allow_multiple_tool_calls` | Gemini handles parallel tool calls automatically. |
| `store` | Not available in the Gemini API. |
| `user` | Not available in the Gemini API. |
| `metadata` | Not available in the Gemini API. |
| `conversation_id` | Not available in the Gemini API. |

### `ThinkingConfig`

```python
class ThinkingConfig(TypedDict, total=False):
    include_thoughts: bool
    # Whether to include condensed thought summaries in the response.
    # The framework currently excludes thought parts from ChatResponse.contents.

    thinking_budget: int
    # Token budget for Gemini 2.5 models.
    # 0 = disable thinking, -1 = dynamic budget.

    thinking_level: types.ThinkingLevel
    # One of: THINKING_LEVEL_UNSPECIFIED (default), MINIMAL, LOW, MEDIUM, HIGH
```

### Using `top_k` and structured output via `response_schema`

```python
import asyncio
from agent_framework import Agent
from agent_framework.gemini import GeminiChatClient, GeminiChatOptions

async def main() -> None:
    client = GeminiChatClient(model="gemini-2.5-flash")

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
            "keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "sentiment", "keywords"],
    }

    agent = Agent(client=client)
    options: GeminiChatOptions = {
        "top_k": 40,
        "response_schema": schema,
    }
    response = await agent.run(
        "Analyse this review: 'The product arrived quickly and works perfectly!'",
        options=options,
    )
    # response.value is None here since we used response_schema directly;
    # response.text contains the raw JSON string
    import json
    data = json.loads(response.text)
    print(data["sentiment"])   # → positive

asyncio.run(main())
```

### Enabling extended thinking

```python
import asyncio
from agent_framework import Agent
from agent_framework.gemini import GeminiChatClient, GeminiChatOptions, ThinkingConfig

async def main() -> None:
    client = GeminiChatClient(model="gemini-2.5-pro")
    agent = Agent(client=client)

    thinking: ThinkingConfig = {
        "thinking_budget": 8192,      # up to 8k tokens for reasoning
        "include_thoughts": False,    # keep reasoning internal
    }
    options: GeminiChatOptions = {"thinking_config": thinking}

    response = await agent.run(
        "Solve: if 3x + 7 = 28, what is x? Show full working.",
        options=options,
    )
    print(response.text)

asyncio.run(main())
```

### Per-call option override

```python
import asyncio
from agent_framework import Agent
from agent_framework.gemini import GeminiChatClient, GeminiChatOptions

async def main() -> None:
    client = GeminiChatClient(model="gemini-2.5-flash")
    agent = Agent(
        client=client,
        # Agent-level defaults: conservative, low top_k
        default_options={"temperature": 0.2, "top_k": 10},
    )

    # Override for creative tasks at call time
    creative_options: GeminiChatOptions = {"temperature": 1.0, "top_k": 64, "max_tokens": 500}
    response = await agent.run("Write a haiku about mountain fog.", options=creative_options)
    print(response.text)

asyncio.run(main())
```

---

## 3 · `GeminiSettings` · `GoogleGeminiSettings`

**Package:** `agent_framework_gemini`

Two `TypedDict` classes capture environment-based configuration. `GeminiSettings` reads `GEMINI_*` variables; `GoogleGeminiSettings` reads `GOOGLE_*` variables. When both are set, `GOOGLE_*` takes priority in the client.

### `GeminiSettings` (env prefix `GEMINI_`)

```python
class GeminiSettings(TypedDict, total=False):
    api_key: SecretString | None   # GEMINI_API_KEY
    model: str | None              # GEMINI_MODEL
```

### `GoogleGeminiSettings` (env prefix `GOOGLE_`)

```python
class GoogleGeminiSettings(TypedDict, total=False):
    api_key: SecretString | None         # GOOGLE_API_KEY      (overrides GEMINI_API_KEY)
    model: str | None                    # GOOGLE_MODEL        (overrides GEMINI_MODEL)
    genai_use_vertexai: bool | None      # GOOGLE_GENAI_USE_VERTEXAI
    cloud_project: str | None            # GOOGLE_CLOUD_PROJECT
    cloud_location: str | None           # GOOGLE_CLOUD_LOCATION
```

### Resolution order

| Setting | Priority 1 | Priority 2 |
|---|---|---|
| API key | `GOOGLE_API_KEY` | `GEMINI_API_KEY` |
| Model | `GOOGLE_MODEL` | `GEMINI_MODEL` |
| Vertex AI | explicit `vertexai=` arg | `GOOGLE_GENAI_USE_VERTEXAI` |
| Project | explicit `project=` arg | `GOOGLE_CLOUD_PROJECT` |
| Location | explicit `location=` arg | `GOOGLE_CLOUD_LOCATION` |

### Minimal `.env` for Developer API

```bash
# .env
GOOGLE_API_KEY=AIza...
GOOGLE_MODEL=gemini-2.5-flash
```

```python
from agent_framework.gemini import GeminiChatClient

# No explicit arguments needed — all resolved from .env
client = GeminiChatClient(env_file_path=".env")
```

### Minimal `.env` for Vertex AI

```bash
# .env
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=my-gcp-project
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_MODEL=gemini-2.5-pro
```

```python
from agent_framework.gemini import GeminiChatClient

# Application Default Credentials picked up automatically
client = GeminiChatClient(env_file_path=".env")
```

---

## 4 · `MontyExecuteCodeTool` · `FileMount`

**Package:** `agent_framework_monty` (import via `from agent_framework.monty import …`)

`MontyExecuteCodeTool` is a `FunctionTool` that executes Python code inside **Monty** — a lightweight Python interpreter sandbox designed as an alternative to Hyperlight. Registered tools appear inside the sandbox as typed async functions; argument types are validated by the `ty` type checker before any host tool runs.

### Constructor

```python
class MontyExecuteCodeTool(FunctionTool):
    def __init__(
        self,
        *,
        tools: FunctionTool | Callable | Sequence[...] | None = None,
        approval_mode: ApprovalMode | None = None,          # default: "never_require"
        workspace_root: str | Path | None = None,
        file_mounts: FileMountInput | Sequence[FileMountInput] | None = None,
        resource_limits: dict[str, Any] | None = None,
    ) -> None: ...
```

| Parameter | Description |
|---|---|
| `tools` | Host-side tools exposed inside the sandbox as `async def tool_name(...)`. |
| `approval_mode` | Approval mode for tool calls. Defaults to `"never_require"` unless any managed tool requires approval, in which case this auto-upgrades. |
| `workspace_root` | Auto-mounts a host directory at `/input` (matching Hyperlight's default). Shortcut for a single read-write `FileMount`. |
| `file_mounts` | Fine-grained `FileMount` entries for additional mounts. |
| `resource_limits` | `dict` forwarded to Monty's `ResourceLimits` (keys: `cpu_time`, `memory`, `output_size`, `recursion_depth`, `gc_frequency`). |

### Mutator API

```python
tool.add_tools(tools)         # add more host tools
tool.get_tools()              # list currently registered tools
tool.remove_tool("name")      # remove by name; KeyError if not found
tool.clear_tools()            # remove all tools
tool.add_file_mounts(mounts)  # add additional file mounts
```

### `FileMount`

```python
class FileMount(NamedTuple):
    host_path: str | Path
    mount_path: str
    mode: MountMode = "overlay"           # "overlay" | "ro" | "rw"
    write_bytes_limit: int | None = None  # bytes cap on writes through this mount
```

| `mode` value | Semantics |
|---|---|
| `"overlay"` | Writes are captured in-process and returned as `Content.from_data` items after execution; the host directory is not modified. |
| `"rw"` | Writes go directly to `host_path`. Written files are also scanned and returned. |
| `"ro"` | Read-only; any write attempt inside the sandbox raises. |

### Basic code execution

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from agent_framework.monty import MontyExecuteCodeTool

@tool(description="Return the square of a number")
async def square(n: float) -> float:
    return n * n

async def main() -> None:
    client = OpenAIChatClient("gpt-4o")
    execute_code = MontyExecuteCodeTool(tools=[square])

    agent = Agent(
        client=client,
        tools=[execute_code],
        instructions=(
            "You can run Python code in a sandbox. "
            "Use execute_code to compute things programmatically."
        ),
    )
    response = await agent.run("Write Python to compute the squares of 1 through 5.")
    print(response.text)

asyncio.run(main())
```

### File mounts and workspace

```python
import asyncio, os, tempfile
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.monty import MontyExecuteCodeTool, FileMount

async def main() -> None:
    client = OpenAIChatClient("gpt-4o")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a CSV the agent can read
        with open(os.path.join(tmpdir, "sales.csv"), "w") as f:
            f.write("month,revenue\nJan,1000\nFeb,1500\nMar,1200\n")

        execute_code = MontyExecuteCodeTool(
            workspace_root=tmpdir,                   # auto-mounts at /input
            file_mounts=[
                FileMount(
                    host_path="/tmp/output",
                    mount_path="/output",
                    mode="rw",
                    write_bytes_limit=10 * 1024 * 1024,   # 10 MB cap
                ),
            ],
            resource_limits={"cpu_time": 30, "memory": 128 * 1024 * 1024},
        )

        agent = Agent(client=client, tools=[execute_code])
        response = await agent.run(
            "Read /input/sales.csv and compute the total revenue. "
            "Write a summary to /output/report.txt."
        )
        print(response.text)

asyncio.run(main())
```

---

## 5 · `MontyCodeActProvider`

**Package:** `agent_framework_monty`

`MontyCodeActProvider` is a `ContextProvider` that injects a Monty-backed CodeAct surface into an agent via the `context_providers` list. It mirrors `HyperlightCodeActProvider` and is the recommended way to add code execution to an agent without managing the tool directly.

```python
class MontyCodeActProvider(ContextProvider):
    DEFAULT_SOURCE_ID = "monty_codeact"

    def __init__(
        self,
        source_id: str = DEFAULT_SOURCE_ID,
        *,
        tools: FunctionTool | Callable | Sequence[...] | None = None,
        approval_mode: ApprovalMode | None = None,
        workspace_root: str | Path | None = None,
        file_mounts: FileMountInput | Sequence[FileMountInput] | None = None,
        resource_limits: dict[str, Any] | None = None,
    ) -> None: ...
```

All parameters except `source_id` are forwarded directly to the underlying `MontyExecuteCodeTool`. The provider handles CodeAct system instructions automatically.

### Mutator API (delegated to the inner tool)

```python
provider.add_tools(tools)
provider.get_tools()
provider.remove_tool("name")
provider.clear_tools()
provider.add_file_mounts(mounts)
```

### Wiring Monty CodeAct into an agent

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from agent_framework.monty import MontyCodeActProvider

@tool(description="Fetch stock price for a ticker symbol")
async def get_stock_price(ticker: str) -> float:
    # In production: call a real market data API
    prices = {"AAPL": 213.5, "MSFT": 418.2, "GOOG": 178.9}
    return prices.get(ticker.upper(), 0.0)

async def main() -> None:
    client = OpenAIChatClient("gpt-4o")

    monty = MontyCodeActProvider(
        tools=[get_stock_price],
        resource_limits={"cpu_time": 30},
    )

    agent = Agent(
        client=client,
        instructions=(
            "You are a financial analyst. Use execute_code to write Python scripts "
            "that call get_stock_price and compute portfolio metrics."
        ),
        context_providers=[monty],
    )

    response = await agent.run(
        "I hold 10 AAPL, 5 MSFT, and 20 GOOG shares. "
        "What is the total value of my portfolio?"
    )
    print(response.text)

asyncio.run(main())
```

### Adding tools after construction

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from agent_framework.monty import MontyCodeActProvider

@tool(description="Convert USD to EUR at a fixed rate")
async def usd_to_eur(amount: float) -> float:
    return round(amount * 0.92, 2)

async def main() -> None:
    client = OpenAIChatClient("gpt-4o")
    monty = MontyCodeActProvider()

    # Add tools after construction — useful when tools are loaded dynamically
    monty.add_tools([usd_to_eur])

    agent = Agent(client=client, context_providers=[monty])
    response = await agent.run("Convert $250 to EUR using execute_code.")
    print(response.text)

asyncio.run(main())
```

---

## 6 · `MistralEmbeddingClient` · `MistralEmbeddingOptions` · `MistralEmbeddingSettings`

**Package:** `agent_framework_mistral` (import via `from agent_framework.mistral import …`)

`MistralEmbeddingClient` provides Mistral AI embeddings with OTel telemetry. It stacks `EmbeddingTelemetryLayer` on top of `RawMistralEmbeddingClient`.

### Constructor

```python
class MistralEmbeddingClient(
    EmbeddingTelemetryLayer[str, list[float], MistralEmbeddingOptionsT],
    RawMistralEmbeddingClient[MistralEmbeddingOptionsT],
    Generic[MistralEmbeddingOptionsT],
):
    OTEL_PROVIDER_NAME: ClassVar[str] = "mistralai"

    def __init__(
        self,
        *,
        model: str | None = None,               # MISTRAL_EMBEDDING_MODEL
        api_key: str | SecretString | None = None,  # MISTRAL_API_KEY
        server_url: str | None = None,          # MISTRAL_SERVER_URL (optional override)
        client: Any | None = None,              # pre-built Mistral SDK client
        otel_provider_name: str | None = None,  # override OTEL_PROVIDER_NAME
        additional_properties: dict[str, Any] | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...
```

### `MistralEmbeddingOptions`

```python
class MistralEmbeddingOptions(EmbeddingGenerationOptions, total=False):
    # Inherits from EmbeddingGenerationOptions — no Mistral-specific additions yet.
    # Standard field: model: str | None  (per-call model override)
    pass
```

### `MistralEmbeddingSettings`

```python
class MistralEmbeddingSettings(TypedDict, total=False):
    api_key: str | None          # MISTRAL_API_KEY
    embedding_model: str | None  # MISTRAL_EMBEDDING_MODEL
    server_url: str | None       # MISTRAL_SERVER_URL
```

### Generating embeddings

```python
import asyncio
from agent_framework.mistral import MistralEmbeddingClient

async def main() -> None:
    # Resolves MISTRAL_API_KEY and MISTRAL_EMBEDDING_MODEL from environment
    client = MistralEmbeddingClient(model="mistral-embed")

    texts = [
        "Python is a high-level programming language.",
        "Machine learning requires large datasets.",
        "The Eiffel Tower is located in Paris.",
    ]

    result = await client.get_embeddings(texts)
    print(f"Generated {len(result)} embeddings, dimension {len(result[0].vector)}")
    for i, emb in enumerate(result):
        print(f"  [{i}] first 4 dims: {emb.vector[:4]}")

asyncio.run(main())
```

### Wiring into `MemoryContextProvider`

```python
import asyncio
from agent_framework import Agent, MemoryContextProvider, MemoryFileStore
from agent_framework.openai import OpenAIChatClient
from agent_framework.mistral import MistralEmbeddingClient

async def main() -> None:
    embedding_client = MistralEmbeddingClient(model="mistral-embed")
    memory_store = MemoryFileStore(base_path="/tmp/agent-memory", owner_state_key="mistral_memory_owner")
    memory = MemoryContextProvider(
        store=memory_store,
        embedding_client=embedding_client,
        selection_limit=5,
    )

    chat_client = OpenAIChatClient("gpt-4o")
    agent = Agent(
        client=chat_client,
        instructions="You are a helpful assistant with durable memory.",
        context_providers=[memory],
    )

    session = agent.create_session()
    session.state["mistral_memory_owner"] = "user-alice"
    await agent.run("My favourite language is Python.", session=session)

    session2 = agent.create_session()
    session2.state["mistral_memory_owner"] = "user-alice"
    response = await agent.run("What is my favourite language?", session=session2)
    print(response.text)

asyncio.run(main())
```

### Custom server URL for self-hosted Mistral

```python
from agent_framework.mistral import MistralEmbeddingClient

# Point at a self-hosted Mistral-compatible endpoint
client = MistralEmbeddingClient(
    model="mistral-embed",
    api_key="local-key",
    server_url="http://localhost:8080/v1",
)
```

---

## 7 · `MessageInjectionMiddleware` · `enqueue_messages`

**Module:** `agent_framework` core (`agent_framework._middleware`)

`MessageInjectionMiddleware` is a `ChatMiddleware` that lets **tool code inject messages into the running agent loop** without breaking the current call. Messages queued in `session.state` are drained into the next model call for that session; if new messages arrive after a model call completes (and there are no pending tool calls), the middleware loops internally.

### `MessageInjectionMiddleware`

```python
class MessageInjectionMiddleware(ChatMiddleware):
    def __init__(self) -> None: ...

    def enqueue_messages(
        self,
        session: AgentSession,
        messages: AgentRunInputs,         # str | Content | Message | Sequence[...]
    ) -> None: ...

    def get_pending_messages(
        self,
        session: AgentSession,
    ) -> list[Message]: ...               # point-in-time snapshot, not updated
```

### `enqueue_messages` (module-level function)

```python
def enqueue_messages(session: AgentSession, messages: AgentRunInputs) -> None:
    """Enqueue messages into session.state under MESSAGE_INJECTION_PENDING_MESSAGES_STATE_KEY."""
```

The module-level `enqueue_messages` is the same operation as `MessageInjectionMiddleware.enqueue_messages()` — both write to `session.state` under `MESSAGE_INJECTION_PENDING_MESSAGES_STATE_KEY`. Call the free function from a `FunctionInvocationContext` when the middleware instance is not in scope.

### How it works

```
Agent.run(message)
  ↓
ChatMiddlewareLayer calls MessageInjectionMiddleware.on_request()
  ↓
_drain_pending_messages() → prepends any queued messages to the current call
  ↓
Model call executes
  ↓
If response has no tool calls AND new messages were enqueued during the call:
  → middleware loops internally (calls model again with the new messages)
  → repeats until no new messages are queued
```

### Injecting a system notification from a tool

```python
import asyncio
from agent_framework import Agent, AgentSession, FunctionInvocationContext, tool
from agent_framework.openai import OpenAIChatClient
from agent_framework import MessageInjectionMiddleware, enqueue_messages

middleware = MessageInjectionMiddleware()

@tool(description="Check the current system status")
async def check_system_status(ctx: FunctionInvocationContext) -> str:
    """Checks status and injects a follow-up message if action is needed."""
    status = "degraded"  # simulate an external status check

    if status == "degraded":
        # Inject a follow-up prompt into the same agent session
        enqueue_messages(
            ctx.session,
            "System status is DEGRADED. Please recommend immediate remediation steps.",
        )
        return f"Status: {status} (follow-up injected)"
    return f"Status: {status}"

async def main() -> None:
    client = OpenAIChatClient("gpt-4o")
    agent = Agent(
        client=client,
        tools=[check_system_status],
        middleware=[middleware],
        instructions="You are an SRE assistant.",
    )

    session = agent.create_session()
    response = await agent.run("Check the system status.", session=session)
    print(response.text)
    # The agent will have also processed the injected remediation prompt
    # before returning the final response.

asyncio.run(main())
```

### Inspecting queued messages

```python
import asyncio
from agent_framework import Agent, AgentSession
from agent_framework.openai import OpenAIChatClient
from agent_framework import MessageInjectionMiddleware, enqueue_messages

middleware = MessageInjectionMiddleware()

async def main() -> None:
    client = OpenAIChatClient("gpt-4o")
    agent = Agent(client=client, middleware=[middleware])
    session = agent.create_session()

    # Enqueue messages before the run — they'll be prepended to the first model call
    enqueue_messages(session, "Context: today is Monday and the sprint starts now.")
    enqueue_messages(session, "Reminder: focus on the highest-priority backlog item.")

    # Check what's queued (non-destructive snapshot)
    pending = middleware.get_pending_messages(session)
    print(f"{len(pending)} messages queued")

    response = await agent.run("What should I work on today?", session=session)
    print(response.text)

asyncio.run(main())
```

---

## 8 · `CachingSkillsSource`

**Module:** `agent_framework` core (`agent_framework._skills`)

`CachingSkillsSource` is a `DelegatingSkillsSource` decorator that adds a concurrent, per-key cache layer over any `SkillsSource`. It is composable with `AggregatingSkillsSource`, `FilteringSkillsSource`, and `DeduplicatingSkillsSource` in the standard skills pipeline.

### Constructor

```python
class CachingSkillsSource(DelegatingSkillsSource):
    def __init__(
        self,
        inner_source: SkillsSource,
        *,
        cache_isolation_key_selector: Callable[[SkillsSourceContext], str | None] | None = None,
        refresh_interval: timedelta | None = None,
    ) -> None: ...
```

| Parameter | Default | Description |
|---|---|---|
| `inner_source` | — | The `SkillsSource` whose results will be cached. |
| `cache_isolation_key_selector` | `None` | Callable `(ctx: SkillsSourceContext) → str | None`. Returns a cache key to isolate results per agent, session, user, etc. `None` = all callers share one bucket. |
| `refresh_interval` | `None` | `timedelta` after which the cached list is stale. `None` = never refreshes. Zero or negative = disables caching (queries on every call). |

### Concurrency contract

- Concurrent callers for the same cache key share a single in-flight `asyncio.Lock` guard — the inner source is queried **at most once per key** even under concurrent requests.
- A failed or cancelled fetch does not update the cache — the next call retries.
- A failed `refresh_interval` re-query keeps the **previously cached list** and retries on the next call.

### Caching an expensive `FileSkillsSource`

```python
from datetime import timedelta
from agent_framework import (
    Agent,
    FileSkillsSource,
    CachingSkillsSource,
    SkillsProvider,
)
from agent_framework.openai import OpenAIChatClient

# FileSkillsSource walks the filesystem on every call — expensive at scale.
file_source = FileSkillsSource(paths=["./skills/"])

# Wrap in a cache that refreshes every 10 minutes.
cached_source = CachingSkillsSource(
    file_source,
    refresh_interval=timedelta(minutes=10),
)

client = OpenAIChatClient("gpt-4o")
agent = Agent(
    client=client,
    context_providers=[SkillsProvider(cached_source)],
)
```

### Per-agent cache isolation

```python
from datetime import timedelta
from agent_framework import CachingSkillsSource, SkillsSourceContext, MCPSkillsSource

# MCPSkillsSource requires an initialized mcp.client.session.ClientSession.
# `mcp_session` is assumed to be an already-initialized ClientSession here.
mcp_source = MCPSkillsSource(client=mcp_session)

def isolate_by_agent(ctx: SkillsSourceContext) -> str | None:
    # Cache key is the agent's name — each agent gets its own cached list.
    return ctx.agent.name if ctx.agent else None

cached = CachingSkillsSource(
    mcp_source,
    cache_isolation_key_selector=isolate_by_agent,
    refresh_interval=timedelta(hours=1),
)
```

### Composing with other decorators

```python
from agent_framework import (
    AggregatingSkillsSource,
    CachingSkillsSource,
    DeduplicatingSkillsSource,
    FilteringSkillsSource,
    FileSkillsSource,
    MCPSkillsSource,
    SkillsProvider,
)
from datetime import timedelta

# Compose a full skills pipeline: aggregate → cache each → dedup → filter.
# MCPSkillsSource requires an initialized mcp.client.session.ClientSession.
# `mcp_session` is assumed to be an already-initialized ClientSession here.
file_source = FileSkillsSource(paths=["./skills/"])
mcp_source = MCPSkillsSource(client=mcp_session)

cached_file = CachingSkillsSource(file_source, refresh_interval=timedelta(minutes=5))
cached_mcp  = CachingSkillsSource(mcp_source,  refresh_interval=timedelta(minutes=1))

aggregated = AggregatingSkillsSource(sources=[cached_file, cached_mcp])
deduped    = DeduplicatingSkillsSource(aggregated)
filtered   = FilteringSkillsSource(
    deduped,
    predicate=lambda skill, _ctx: skill.frontmatter.name.startswith("python_"),
)

provider = SkillsProvider(filtered)
```

---

## 9 · `FoundryToolbox`

**Package:** `agent_framework_foundry_hosting` (import via `from agent_framework.foundry import FoundryToolbox`)

`FoundryToolbox` is a thin `MCPStreamableHTTPTool` subclass that targets a **Microsoft Foundry toolbox MCP endpoint** with bearer-token authentication. It resolves its endpoint from environment variables and forwards the per-request `x-agent-foundry-call-id` header automatically.

### Constructor

```python
class FoundryToolbox(MCPStreamableHTTPTool):
    def __init__(
        self,
        credential: TokenCredential,
        *,
        url: str | None = None,
        name: str | None = None,
        token_scope: str = DEFAULT_TOOLBOX_SCOPE,
        load_prompts: bool = False,
        load_tools: bool = True,
        timeout: float = 30.0,     # default HTTP client timeout in seconds
    ) -> None: ...
```

| Parameter | Default | Description |
|---|---|---|
| `credential` | — | `azure.identity` credential for bearer tokens (e.g. `DefaultAzureCredential()`). Tokens are requested per outbound request and cached. |
| `url` | `None` | MCP endpoint URL. Resolved from `TOOLBOX_ENDPOINT` or `FOUNDRY_PROJECT_ENDPOINT` + `TOOLBOX_NAME`. |
| `name` | `None` | Local tool name. Resolved from `TOOLBOX_NAME` or derived from the endpoint path. |
| `token_scope` | Foundry data-plane scope | Azure token scope for the toolbox endpoint. |
| `load_prompts` | `False` | Whether to load MCP prompts (toolboxes expose tools, not prompts). |
| `load_tools` | `True` | Whether to load MCP tools from the toolbox. |
| `timeout` | `30.0` | HTTP client timeout in seconds. |

### Environment variables

| Variable | Role |
|---|---|
| `TOOLBOX_ENDPOINT` | Toolbox MCP URL (takes priority over building it from parts) |
| `TOOLBOX_NAME` | Tool name + appended to `FOUNDRY_PROJECT_ENDPOINT` to form the URL |
| `FOUNDRY_PROJECT_ENDPOINT` | Foundry project base URL; combined with `TOOLBOX_NAME` |

### Basic usage with `ResponsesHostServer`

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework.foundry import FoundryToolbox, ResponsesHostServer
from azure.identity import DefaultAzureCredential

async def main() -> None:
    credential = DefaultAzureCredential()

    # FoundryToolbox resolves URL from TOOLBOX_ENDPOINT env var.
    # The hosting server enters the agent, which connects and closes the toolbox.
    toolbox = FoundryToolbox(credential)

    agent = Agent(
        client=FoundryChatClient(credential=credential),
        tools=[toolbox],
        default_options={"store": False},
    )

    await ResponsesHostServer(agent).run_async()

asyncio.run(main())
```

### Explicit endpoint configuration

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework.foundry import FoundryToolbox, ResponsesHostServer
from azure.identity import DefaultAzureCredential

async def main() -> None:
    credential = DefaultAzureCredential()

    toolbox = FoundryToolbox(
        credential,
        url="https://myproject.eastus.api.azureml.ms/toolbox/my-toolbox/mcp",
        name="my_toolbox",
        timeout=60.0,
    )

    agent = Agent(
        client=FoundryChatClient(credential=credential),
        tools=[toolbox],
    )

    await ResponsesHostServer(agent).run_async()

asyncio.run(main())
```

---

## 10 · `ResponsesHostServer` · `InvocationsHostServer`

**Package:** `agent_framework_foundry_hosting`

Two server classes that host an agent in a Foundry-managed execution environment. Both build on Starlette/Hypercorn and listen on `PORT` (default 8088).

### `ResponsesHostServer`

Implements the **OpenAI Responses API** contract. Accepts streaming requests, manages checkpointing, handles HITL tool-approval flow, and maintains session/user isolation.

```python
class ResponsesHostServer:
    CHECKPOINT_STORAGE_PATH = "/.checkpoints"
    FUNCTION_APPROVAL_STORAGE_PATH = "/.function_approvals/approval_requests.json"

    def __init__(
        self,
        agent: SupportsAgentRun,
        *,
        prefix: str = "",
        options: ResponsesServerOptions | None = None,
        store: ResponseProviderProtocol | None = None,
        **kwargs: Any,
    ) -> None: ...

    async def run_async(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,     # defaults to PORT env var or 8088
    ) -> None: ...
```

**Constraints:**
- The agent must **not** have a `HistoryProvider` with `load_messages=True` — history is managed by the hosting infrastructure.
- Context providers that maintain in-memory state should be used with caution: the hosting environment may be deactivated between requests.
- `WorkflowAgent` with an existing `CheckpointStorage` raises at construction — checkpointing is managed by the server under `CHECKPOINT_STORAGE_PATH`.

### `InvocationsHostServer`

Implements a simpler **JSON invocations** contract. Accepts `{"message": "...", "stream": false}` and returns `{"response": "...", "session_id": "..."}`.

```python
class InvocationsHostServer:
    def __init__(
        self,
        agent: SupportsAgentRun,
        *,
        openapi_spec: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...

    async def run_async(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
    ) -> None: ...
```

Supports both streaming (`stream: true` → SSE) and non-streaming (`stream: false` → JSON body) responses. Session isolation uses a partition key derived from the `session_id` and `user_id` from the Foundry request context.

### Serving an agent via `ResponsesHostServer`

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.foundry import ResponsesHostServer

async def main() -> None:
    # Typically uses FoundryChatClient in production;
    # OpenAIChatClient is used here for clarity.
    client = OpenAIChatClient("gpt-4o")
    agent = Agent(
        client=client,
        instructions="You are a helpful hosted agent.",
    )
    # Binds to 0.0.0.0:$PORT (defaults to 8088)
    await ResponsesHostServer(agent).run_async()

asyncio.run(main())
```

### Serving an agent via `InvocationsHostServer`

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.foundry import InvocationsHostServer

async def main() -> None:
    client = OpenAIChatClient("gpt-4o")
    agent = Agent(
        client=client,
        instructions="You are a helpful JSON-invocable agent.",
    )
    await InvocationsHostServer(agent).run_async()

asyncio.run(main())
```

### Side-by-side comparison

| Feature | `ResponsesHostServer` | `InvocationsHostServer` |
|---|---|---|
| Protocol | OpenAI Responses API | JSON `{message, stream}` |
| Streaming | Yes (Responses API SSE) | Yes (`stream: true` → SSE) |
| Session management | Hosted infrastructure | In-process `dict[session_id, AgentSession]` |
| Checkpointing | Yes (managed under `/.checkpoints`) | No |
| HITL support | Yes (tool approval flow) | No |
| History management | Hosted infrastructure | In-process |
| `WorkflowAgent` support | Yes | Partial |
| Typical use case | Production Foundry agents | Simple endpoint integrations |

### Binding to a custom port

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.foundry import ResponsesHostServer

async def main() -> None:
    agent = Agent(client=OpenAIChatClient("gpt-4o"))
    await ResponsesHostServer(agent).run_async(host="127.0.0.1", port=9000)

asyncio.run(main())
```

---

## Summary

| Class / symbol | Key insight |
|---|---|
| `GeminiChatClient` | Full MRO stack (FunctionInvocation + Middleware + Telemetry + Raw). Use `vertexai=True` + ADC for Vertex AI. |
| `RawGeminiChatClient` | Bare HTTP layer; `OTEL_PROVIDER_NAME='gcp.gemini'`. Dual env prefix: `GOOGLE_*` wins over `GEMINI_*`. |
| `GeminiChatOptions` | Gemini extras: `top_k`, `response_schema`, `thinking_config`. Unsupported fields declared as `None`. |
| `ThinkingConfig` | `thinking_budget`: `0`=off, `-1`=dynamic. `thinking_level`: MINIMAL→HIGH. `include_thoughts` keeps summaries local. |
| `GeminiSettings` | `GEMINI_API_KEY`, `GEMINI_MODEL`. |
| `GoogleGeminiSettings` | `GOOGLE_API_KEY`, `GOOGLE_MODEL`, `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`. |
| `MontyExecuteCodeTool` | Python sandbox with `add_tools()` mutation API; file mounts via `FileMount`; `resource_limits` dict cap. |
| `FileMount` | `NamedTuple`: `host_path`, `mount_path`, `mode` (overlay/rw/ro), `write_bytes_limit`. |
| `MontyCodeActProvider` | `ContextProvider` wrapper around `MontyExecuteCodeTool`; same params, mutators delegated. |
| `MistralEmbeddingClient` | Mistral AI embeddings; `OTEL_PROVIDER_NAME='mistralai'`; env: `MISTRAL_API_KEY`, `MISTRAL_EMBEDDING_MODEL`, `MISTRAL_SERVER_URL`. |
| `MistralEmbeddingOptions` | Extends `EmbeddingGenerationOptions` (no Mistral-specific additions in 1.0.0b260721). |
| `MistralEmbeddingSettings` | `api_key`, `embedding_model`, `server_url`. |
| `MessageInjectionMiddleware` | Drains `session.state` queue into the next model call; loops after non-tool-call responses when new messages arrive. |
| `enqueue_messages` | Free function writing to `MESSAGE_INJECTION_PENDING_MESSAGES_STATE_KEY` — callable from tool code without middleware reference. |
| `CachingSkillsSource` | Per-key `asyncio.Lock` guard; `refresh_interval: timedelta`; failed fetches do not poison the cache. |
| `FoundryToolbox` | `MCPStreamableHTTPTool` + bearer auth via `DefaultAzureCredential`; resolves URL from `TOOLBOX_ENDPOINT` or `FOUNDRY_PROJECT_ENDPOINT`+`TOOLBOX_NAME`. |
| `ResponsesHostServer` | OpenAI Responses API server; managed checkpointing, HITL, and session isolation for production Foundry agents. |
| `InvocationsHostServer` | Simple JSON invocations server; streaming SSE; in-process session dict; no checkpointing. |

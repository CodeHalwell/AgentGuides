---
title: "Semantic Kernel Comprehensive Guide (Python)"
description: "Semantic Kernel for Python — kernel and services, prompt and native functions, plugins, function calling, agents, orchestrations, vector stores, MCP, filters, processes and migration to Microsoft Agent Framework."
framework: semantic-kernel
language: python
---

# Semantic Kernel Comprehensive Guide (Python)

> Verified against `semantic-kernel` %%version=semantic-kernel%% (checked %%versions-checked%%). Imports, constructors and method signatures on this page were checked against an installed copy; the agent, orchestration, filter and vector-store examples were run end to end against a stub chat service.

> **Semantic Kernel and Microsoft Agent Framework.** [Microsoft Agent Framework](/microsoft-agent-framework-guide/) (MAF) is the successor to Semantic Kernel, built by the same team, and reached 1.0 (general availability) in April 2026. Microsoft continues to support Semantic Kernel 1.x — critical bugs, security fixes, and some features taken to GA — for at least a year after MAF's GA, but most new features land in MAF. Start new agent projects on MAF; keep existing SK code running and migrate at your own pace ([section 16](#16-migrating-to-microsoft-agent-framework)).

---

## Table of contents

1. [Installation and setup](#1-installation-and-setup)
2. [The kernel](#2-the-kernel)
3. [AI services](#3-ai-services)
4. [Prompt functions](#4-prompt-functions)
5. [Native functions](#5-native-functions)
6. [Plugins](#6-plugins)
7. [Function calling (and what happened to planners)](#7-function-calling-and-what-happened-to-planners)
8. [Agents](#8-agents)
9. [Multi-agent orchestration](#9-multi-agent-orchestration)
10. [Chat history and memory](#10-chat-history-and-memory)
11. [Vector stores](#11-vector-stores)
12. [Structured output](#12-structured-output)
13. [Model Context Protocol (MCP)](#13-model-context-protocol-mcp)
14. [Filters](#14-filters)
15. [Process framework](#15-process-framework)
16. [Migrating to Microsoft Agent Framework](#16-migrating-to-microsoft-agent-framework)
17. [Removed and renamed APIs](#17-removed-and-renamed-apis)

---

## 1. Installation and setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install "semantic-kernel==%%version=semantic-kernel%%"
```

Python 3.10 or newer is required. A plain `pip install` works: the package depends on a pre-release of `azure-ai-agents`, but because the requirement itself names a pre-release, pip resolves it without `--pre`.

**Optional extras** pull in provider SDKs:

```bash
pip install "semantic-kernel[azure]"        # Azure AI Search, Cosmos DB, Azure AI Inference ...
pip install "semantic-kernel[anthropic]"    # Anthropic connector
pip install "semantic-kernel[google]"       # Google AI / Vertex AI connectors
pip install "semantic-kernel[ollama]"       # Ollama connector
pip install "semantic-kernel[qdrant]"       # Qdrant vector store (similarly: redis, chroma, postgres, ...)
```

The multi-agent orchestrations (section 9) use an in-process runtime that imports `protobuf`; if `from semantic_kernel.agents.runtime import InProcessRuntime` fails with `No module named 'google'`, run `pip install protobuf`.

**Configuration.** Connectors read settings from the environment (or a `.env` file passed as `env_file_path=`):

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL_ID=gpt-5-mini
OPENAI_EMBEDDING_MODEL_ID=text-embedding-3-small

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=<deployment>
AZURE_OPENAI_API_KEY=...          # omit to use Entra ID (see below)
```

Model and deployment names in this guide are examples.

---

## 2. The kernel

The `Kernel` is a container for AI services, plugins and filters. Functions run *through* a kernel, which supplies the services they need.

```python
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id="gpt-5-mini"))

async def main():
    result = await kernel.invoke_prompt("Give me one sentence about Semantic Kernel.")
    print(result)

asyncio.run(main())
```

Kernels are cheap. A common pattern is one kernel per request (or per agent) built from shared service instances, so that per-request plugins and filters do not leak between users.

---

## 3. AI services

### 3.1 OpenAI and Azure OpenAI

```python
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion, OpenAITextEmbedding,
    AzureChatCompletion, AzureTextEmbedding,
)

# OpenAI (reads OPENAI_API_KEY / OPENAI_CHAT_MODEL_ID if arguments are omitted)
kernel.add_service(OpenAIChatCompletion(service_id="openai", ai_model_id="gpt-5-mini"))
kernel.add_service(OpenAITextEmbedding(service_id="embed", ai_model_id="text-embedding-3-small"))

# Azure OpenAI with an API key
kernel.add_service(AzureChatCompletion(
    service_id="azure",
    deployment_name="gpt-5-mini",
    endpoint="https://my-resource.openai.azure.com/",
    api_key="...",
))
```

### 3.2 Entra ID (managed identity) for Azure

```python
from azure.identity import DefaultAzureCredential

kernel.add_service(AzureChatCompletion(
    service_id="azure",
    deployment_name="gpt-5-mini",
    endpoint="https://my-resource.openai.azure.com/",
    credential=DefaultAzureCredential(),
))
```

### 3.3 Other providers

Connectors live under `semantic_kernel.connectors.ai.*`: `anthropic.AnthropicChatCompletion`, `google.google_ai.GoogleAIChatCompletion`, `google.vertex_ai.VertexAIChatCompletion`, `ollama.OllamaChatCompletion`, `mistral_ai.MistralAIChatCompletion`, `bedrock.BedrockChatCompletion`, `azure_ai_inference.AzureAIInferenceChatCompletion`, `hugging_face.HuggingFaceTextCompletion` and others. Each needs its extra (section 1).

### 3.4 Several services in one kernel

```python
kernel.add_service(OpenAIChatCompletion(service_id="fast", ai_model_id="gpt-5-nano"))
kernel.add_service(OpenAIChatCompletion(service_id="smart", ai_model_id="gpt-5"))

from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments

result = await kernel.invoke_prompt(
    "Classify the sentiment of: {{$text}}",
    arguments=KernelArguments(
        text="I love it",
        settings=OpenAIChatPromptExecutionSettings(service_id="fast"),
    ),
)
```

The `service_id` on the execution settings selects the service; without one, the kernel uses its default.

### 3.5 Calling a chat service directly

```python
from semantic_kernel.contents import ChatHistory

chat = kernel.get_service("openai")
history = ChatHistory(system_message="You are terse.")
history.add_user_message("What is RAG?")

reply = await chat.get_chat_message_content(history, OpenAIChatPromptExecutionSettings())
print(reply.content)
```

---

## 4. Prompt functions

### 4.1 Inline prompt functions

```python
from semantic_kernel.functions import KernelArguments

summarise = kernel.add_function(
    plugin_name="writer",
    function_name="summarise",
    prompt="Summarise the following in {{$sentences}} sentences:\n\n{{$input}}",
)

result = await kernel.invoke(summarise, KernelArguments(input=long_text, sentences="2"))
print(result)
```

`{{$variable}}` reads an argument; `{{plugin.function $arg}}` calls another function from inside a template.

### 4.2 Execution settings

```python
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

settings = OpenAIChatPromptExecutionSettings(max_completion_tokens=400)

tagline = kernel.add_function(
    plugin_name="marketing",
    function_name="tagline",
    prompt="Write a tagline for {{$product}}.",
    prompt_execution_settings=settings,
)
```

### 4.3 Template formats

Prompt templates can use the default Semantic Kernel syntax, Handlebars or Jinja2:

```python
kernel.add_function(
    plugin_name="writer",
    function_name="bullets",
    prompt="{% for item in items %}- {{ item }}\n{% endfor %}Rewrite these as prose.",
    template_format="jinja2",
)
```

### 4.4 Prompt files

A plugin directory with one folder per function, each containing `skprompt.txt` and `config.json` (or a single YAML file), loads with:

```python
kernel.add_plugin(parent_directory="./prompts", plugin_name="WriterPlugin")
```

---

## 5. Native functions

Decorate Python methods with `@kernel_function`. Type hints and `Annotated` descriptions become the schema the model sees.

```python
from typing import Annotated
from semantic_kernel.functions import kernel_function

class WeatherPlugin:
    @kernel_function(description="Get the current weather for a city.")
    def get_weather(self, city: Annotated[str, "City name, e.g. 'Paris'"]) -> Annotated[str, "Weather summary"]:
        return f"Sunny, 22°C in {city}"

    @kernel_function(name="get_forecast", description="Get a multi-day forecast.")
    async def forecast(self, city: str, days: int = 3) -> list[str]:
        return [f"Day {i + 1}: sunny" for i in range(days)]

kernel.add_plugin(WeatherPlugin(), plugin_name="weather")

result = await kernel.invoke(plugin_name="weather", function_name="get_weather", city="Paris")
print(result.value)
```

Sync and async methods both work. Keep side-effecting functions small and explicit, because the model decides when to call them.

---

## 6. Plugins

A plugin is a named group of functions. Sources:

| Source | How |
|---|---|
| A class instance with `@kernel_function` methods | `kernel.add_plugin(MyPlugin(), plugin_name="my")` |
| A directory of prompt functions | `kernel.add_plugin(parent_directory=..., plugin_name=...)` |
| An OpenAPI document | `kernel.add_plugin_from_openapi(plugin_name=..., openapi_document_path=...)` |
| An MCP server | `MCPStdioPlugin` / `MCPStreamableHttpPlugin` (section 13) |
| A vector store search | `collection.create_search_function(...)` (section 11) |

**Built-in plugins** (`semantic_kernel.core_plugins`): `TimePlugin`, `MathPlugin`, `TextPlugin`, `HttpPlugin`, `WebSearchEnginePlugin`, `ConversationSummaryPlugin`, `SessionsPythonTool` (Azure Container Apps code interpreter).

**OpenAPI plugins**

```python
from semantic_kernel.connectors.openapi_plugin import OpenAPIFunctionExecutionParameters

kernel.add_plugin_from_openapi(
    plugin_name="petstore",
    openapi_document_path="https://petstore3.swagger.io/api/v3/openapi.json",
    execution_settings=OpenAPIFunctionExecutionParameters(enable_payload_namespacing=True),
)
```

Each operation becomes a kernel function; authentication is supplied through `OpenAPIFunctionExecutionParameters(auth_callback=...)`.

---

## 7. Function calling (and what happened to planners)

Semantic Kernel's old planners (`SequentialPlanner`, `StepwisePlanner`, `ActionPlanner`, and later `FunctionCallingStepwisePlanner`) were removed from the Python package in 1.33.0. Native function calling replaced them: you describe the functions, the model decides which to call, and the kernel executes the calls and feeds back the results until the model answers.

```python
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments

settings = OpenAIChatPromptExecutionSettings(
    function_choice_behavior=FunctionChoiceBehavior.Auto(),
)

result = await kernel.invoke_prompt(
    "What's the weather in Paris and in Rome?",
    arguments=KernelArguments(settings=settings),
)
print(result)
```

| Behaviour | Meaning |
|---|---|
| `FunctionChoiceBehavior.Auto()` | Model may call any advertised function (default for agents) |
| `FunctionChoiceBehavior.Required(filters=...)` | Model must call one of the given functions |
| `FunctionChoiceBehavior.NoneInvoke()` | Functions are advertised but not called — useful for "plan, then confirm" |
| `Auto(filters={"included_plugins": ["weather"]})` | Restrict which functions are advertised |
| `Auto(auto_invoke=False)` | Return the function calls to you instead of executing them |

For multi-step plans that a human must approve, ask the model for a plan with `NoneInvoke()` or structured output (section 12), show it, then execute.

---

## 8. Agents

### 8.1 `ChatCompletionAgent`

```python
import asyncio
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

agent = ChatCompletionAgent(
    service=OpenAIChatCompletion(ai_model_id="gpt-5-mini"),
    name="Assistant",
    instructions="You are a helpful assistant. Use the weather tool for weather questions.",
    plugins=[WeatherPlugin()],
)

async def main():
    response = await agent.get_response(messages="What's the weather in Paris?")
    print(response.message.content)

asyncio.run(main())
```

Pass `service=` directly, or pass `kernel=` (a kernel that already holds the service and plugins). Function calling is on by default for agents.

### 8.2 Threads: multi-turn conversations

```python
from semantic_kernel.agents import ChatHistoryAgentThread

thread: ChatHistoryAgentThread | None = None

for user_input in ["My name is Ada.", "What is my name?"]:
    response = await agent.get_response(messages=user_input, thread=thread)
    thread = response.thread          # created on the first call
    print(response.message.content)

await thread.delete()                  # clean up when the conversation ends
```

### 8.3 Invoke and stream

```python
# All messages the agent produces (including intermediate tool-call messages via the callback)
async def on_intermediate(message):
    print("intermediate:", message.items)

async for item in agent.invoke(messages="Weather in Rome?", on_intermediate_message=on_intermediate):
    print(item.content)

# Token streaming
async for chunk in agent.invoke_stream(messages="Write a haiku about kernels."):
    print(chunk.content, end="", flush=True)
```

### 8.4 Other agent types

| Class | Backed by |
|---|---|
| `ChatCompletionAgent` | Any SK chat completion service |
| `OpenAIAssistantAgent` / `AzureAssistantAgent` | OpenAI Assistants API (being retired by OpenAI; prefer Responses) |
| `OpenAIResponsesAgent` / `AzureResponsesAgent` | OpenAI Responses API |
| `AzureAIAgent` | Azure AI Foundry Agent Service |
| `BedrockAgent` | Amazon Bedrock Agents |
| `CopilotStudioAgent` | Microsoft Copilot Studio |
| `AutoGenConversableAgent` | Wraps an AutoGen 0.2 `ConversableAgent` |

All share `get_response`, `invoke` and `invoke_stream`, and each has a matching thread type.

### 8.5 Declarative agents

`AgentRegistry.create_from_yaml(...)` builds agents from a YAML spec (name, instructions, model, tools). Useful for configuration-driven deployments; the spec format is still marked experimental.

---

## 9. Multi-agent orchestration

Semantic Kernel ships five orchestration patterns sharing one API: build the orchestration, start an `InProcessRuntime`, `invoke` a task, and `await result.get()`. They are marked experimental.

```python
import asyncio
from semantic_kernel.agents import ChatCompletionAgent, SequentialOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

service = OpenAIChatCompletion(ai_model_id="gpt-5-mini")

analyst = ChatCompletionAgent(name="Analyst", description="Extracts key features",
    instructions="List the product's key features and audience.", service=service)
writer = ChatCompletionAgent(name="Writer", description="Writes copy",
    instructions="Write 50 words of marketing copy from the analysis.", service=service)
editor = ChatCompletionAgent(name="Editor", description="Polishes copy",
    instructions="Tighten the copy. Return only the final text.", service=service)

async def main():
    orchestration = SequentialOrchestration(members=[analyst, writer, editor])
    runtime = InProcessRuntime()
    runtime.start()

    result = await orchestration.invoke(task="An eco-friendly steel water bottle.", runtime=runtime)
    final = await result.get(timeout=120)
    print(final.content)

    await runtime.stop_when_idle()

asyncio.run(main())
```

| Pattern | Class | Behaviour |
|---|---|---|
| Sequential | `SequentialOrchestration` | Output of each agent feeds the next |
| Concurrent | `ConcurrentOrchestration` | All agents get the task in parallel; `get()` returns a list |
| Group chat | `GroupChatOrchestration(manager=RoundRobinGroupChatManager(max_rounds=...))` | Agents take turns; a manager picks the speaker and decides when to stop |
| Hand-off | `HandoffOrchestration(handoffs=OrchestrationHandoffs()...)` | Agents transfer control to each other; optional `human_response_function` for human input |
| Magentic | `MagenticOrchestration(manager=StandardMagenticManager(...))` | A manager agent plans and delegates (from the Magentic-One design) |

**Hand-off example**

```python
from semantic_kernel.agents import HandoffOrchestration, OrchestrationHandoffs

handoffs = (
    OrchestrationHandoffs()
    .add(source_agent=triage.name, target_agent=billing.name, description="Billing questions")
    .add(source_agent=triage.name, target_agent=tech.name, description="Technical issues")
    .add(source_agent=billing.name, target_agent=triage.name, description="Anything else")
)

orchestration = HandoffOrchestration(
    members=[triage, billing, tech],
    handoffs=handoffs,
    human_response_function=lambda: ChatMessageContent(role=AuthorRole.USER, content=input("> ")),
)
```

`agent_response_callback=` on any orchestration receives every agent message — use it for logging or a live transcript. `input_transform` / `output_transform` convert between your own types and chat messages.

The older `AgentGroupChat` class (with `SelectionStrategy` / `TerminationStrategy`) still exists but is experimental and superseded by `GroupChatOrchestration`.

---

## 10. Chat history and memory

### 10.1 `ChatHistory` and reducers

```python
from semantic_kernel.contents import ChatHistory, ChatHistoryTruncationReducer, ChatHistorySummarizationReducer

history = ChatHistoryTruncationReducer(target_count=20)        # keeps the last ~20 messages
history.add_system_message("You are helpful.")
history.add_user_message("Hi")
await history.reduce()                                          # trims when over target

summarising = ChatHistorySummarizationReducer(
    service=kernel.get_service("openai"), target_count=20, threshold_count=5,
)
```

Both reducers are `ChatHistory` subclasses, so they can be used wherever a history is expected, including `ChatHistoryAgentThread(chat_history=...)`.

### 10.2 Long-term memory

The old `SemanticTextMemory` / `VolatileMemoryStore` / `TextMemoryPlugin` API is still importable but legacy. For long-term memory, store facts or documents in a **vector store** (section 11) and expose search as a plugin, or use an agent service with built-in memory.

---

## 11. Vector stores

The vector store API was redesigned in 1.34.0: records are plain classes annotated with `VectorStoreField`, collections are generic over the record type, and embeddings can be generated automatically.

```python
from dataclasses import dataclass
from typing import Annotated
from semantic_kernel.data.vector import vectorstoremodel, VectorStoreField
from semantic_kernel.connectors.in_memory import InMemoryCollection
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding

@vectorstoremodel(collection_name="faq")
@dataclass
class FaqEntry:
    id: Annotated[str, VectorStoreField("key")]
    question: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    answer: Annotated[str, VectorStoreField("data")]
    embedding: Annotated[list[float] | str | None, VectorStoreField("vector", dimensions=1536)] = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = f"{self.question} {self.answer}"   # text to embed on upsert

collection = InMemoryCollection(
    record_type=FaqEntry,
    embedding_generator=OpenAITextEmbedding(ai_model_id="text-embedding-3-small"),
)

async def demo():
    await collection.ensure_collection_exists()
    await collection.upsert([
        FaqEntry(id="1", question="How do I reset my password?", answer="Use Settings > Security."),
        FaqEntry(id="2", question="What are your hours?", answer="9–5 UK time, Mon–Fri."),
    ])
    results = await collection.search("forgot password", top=1)
    async for hit in results.results:
        print(hit.score, hit.record.answer)
```

Swap `InMemoryCollection` for another connector without changing the record class:

| Store | Import |
|---|---|
| Azure AI Search | `semantic_kernel.connectors.azure_ai_search.AzureAISearchCollection` |
| Qdrant | `semantic_kernel.connectors.qdrant.QdrantCollection` |
| Postgres (pgvector) | `semantic_kernel.connectors.postgres.PostgresCollection` |
| Redis | `semantic_kernel.connectors.redis.RedisJsonCollection` / `RedisHashsetCollection` |
| Chroma, Pinecone, Weaviate, MongoDB Atlas, Cosmos DB, SQL Server, FAISS | `semantic_kernel.connectors.<name>` |

(`semantic_kernel.connectors.memory` re-exports them all.) Filtering uses Python lambdas, e.g. `collection.search("...", filter=lambda r: r.category == "billing")`, which each connector translates to its native filter syntax. `hybrid_search` is available on stores that support it.

**Search as a plugin (RAG for agents)**

```python
search_fn = collection.create_search_function(
    function_name="search_faq",
    description="Search the FAQ for answers to customer questions.",
)
kernel.add_function(plugin_name="faq", function=search_fn)
```

---

## 12. Structured output

```python
from pydantic import BaseModel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments

class Recipe(BaseModel):
    title: str
    ingredients: list[str]
    minutes: int

settings = OpenAIChatPromptExecutionSettings(response_format=Recipe)

result = await kernel.invoke_prompt(
    "Give me a quick pasta recipe.",
    arguments=KernelArguments(settings=settings),
)
recipe = Recipe.model_validate_json(str(result))
```

For agents, pass the same settings through `arguments=KernelArguments(settings=settings)` on the agent constructor.

---

## 13. Model Context Protocol (MCP)

### 13.1 Using MCP servers as plugins

```python
from semantic_kernel.connectors.mcp import MCPStdioPlugin, MCPStreamableHttpPlugin

async def main():
    async with MCPStdioPlugin(
        name="github",
        description="GitHub tools",
        command="docker",
        args=["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"],
    ) as github, MCPStreamableHttpPlugin(name="docs", url="http://localhost:8000/mcp") as docs:
        agent = ChatCompletionAgent(
            service=OpenAIChatCompletion(ai_model_id="gpt-5-mini"),
            name="DevAssistant",
            instructions="Answer questions about the repository.",
            plugins=[github, docs],
        )
        print(await agent.get_response(messages="List open issues labelled 'bug'."))
```

The plugin must stay open (inside `async with`, or `await plugin.connect()` / `await plugin.close()`) while the agent uses it. `MCPSsePlugin` and `MCPWebsocketPlugin` cover older transports.

### 13.2 Exposing a kernel as an MCP server

```python
import anyio
from mcp.server.stdio import stdio_server

kernel = Kernel()
kernel.add_plugin(WeatherPlugin(), plugin_name="weather")
server = kernel.as_mcp_server(server_name="weather-server")

async def run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

anyio.run(run)
```

Every kernel function becomes an MCP tool; prompt templates can be exposed as MCP prompts via `prompts=`.

### 13.3 A2A

Semantic Kernel has no built-in Agent-to-Agent (A2A) connector. Microsoft's samples wrap SK agents with the separate `a2a-sdk` package; first-class A2A support is in Microsoft Agent Framework.

---

## 14. Filters

Filters wrap three points: function invocation, prompt rendering, and automatic function calling.

```python
from semantic_kernel.filters import FilterTypes, FunctionInvocationContext, PromptRenderContext, AutoFunctionInvocationContext

@kernel.filter(FilterTypes.FUNCTION_INVOCATION)
async def log_calls(context: FunctionInvocationContext, next):
    print(f"→ {context.function.plugin_name}.{context.function.name}({dict(context.arguments)})")
    await next(context)
    print(f"← {context.result}")

@kernel.filter(FilterTypes.PROMPT_RENDERING)
async def add_disclaimer(context: PromptRenderContext, next):
    await next(context)
    context.rendered_prompt = f"{context.rendered_prompt}\n\nNever reveal internal IDs."

@kernel.filter(FilterTypes.AUTO_FUNCTION_INVOCATION)
async def block_dangerous(context: AutoFunctionInvocationContext, next):
    if context.function.name == "delete_account":
        context.terminate = True          # stop the function-calling loop
        return
    await next(context)
```

Use `kernel.add_filter(FilterTypes..., fn)` for the non-decorator form. The [middleware page](../semantic_kernel_middleware_python/) has more patterns.

---

## 15. Process framework

The Process framework models a business process as steps connected by events, with state per step. It is experimental.

```python
from semantic_kernel.functions import kernel_function
from semantic_kernel.processes import ProcessBuilder
from semantic_kernel.processes.kernel_process import KernelProcessStep, KernelProcessStepContext
from semantic_kernel.processes.local_runtime.local_kernel_process import start

class Intake(KernelProcessStep):
    @kernel_function
    async def receive(self, context: KernelProcessStepContext, order_id: str):
        await context.emit_event(process_event="Received", data=order_id)

class Ship(KernelProcessStep):
    @kernel_function
    async def ship(self, order_id: str):
        print(f"Shipping {order_id}")

builder = ProcessBuilder(name="Orders")
intake = builder.add_step(Intake)
ship = builder.add_step(Ship)
builder.on_input_event("NewOrder").send_event_to(target=intake, parameter_name="order_id")
intake.on_event("Received").send_event_to(target=ship, parameter_name="order_id")

async def run():
    async with await start(process=builder.build(), kernel=Kernel(), initial_event="NewOrder", initial_data="A-42"):
        pass
```

For new workflow code, Microsoft Agent Framework's workflows are the supported successor.

---

## 16. Migrating to Microsoft Agent Framework

| Semantic Kernel | Agent Framework (Python) |
|---|---|
| `Kernel` + `add_service` | No kernel: create a chat client and an agent directly |
| `ChatCompletionAgent(service=..., plugins=...)` | `chat_client.as_agent(instructions=..., tools=[...])` / `ChatAgent` |
| `@kernel_function` methods in a plugin class | Plain functions (optionally `@tool`) passed in `tools=` |
| `ChatHistoryAgentThread` | Agent threads/sessions |
| Filters | Middleware |
| Orchestrations / Process framework | Workflows (graph-based, with checkpointing and human-in-the-loop) |

You can migrate incrementally: since SK 1.38, any `KernelFunction` — a prompt function, a native function, or a vector-store search function — converts to an Agent Framework tool:

```python
af_tool = search_fn.as_agent_framework_tool()
```

See Microsoft's [migration guide](https://learn.microsoft.com/agent-framework/migration-guide/from-semantic-kernel/) and this site's [Agent Framework guide](/microsoft-agent-framework-guide/python/) for the target API.

---

## 17. Removed and renamed APIs

| Old | Status | Use instead |
|---|---|---|
| `semantic_kernel.planners` (`SequentialPlanner`, `StepwisePlanner`, `ActionPlanner`, `FunctionCallingStepwisePlanner`) | Removed in 1.33.0 | Function calling (section 7) |
| `VectorStoreRecordCollection`, `VectorStoreRecordKeyField` / `DataField` / `VectorField`, `*MemoryStore` connectors | Replaced in 1.34.0 | `VectorStoreCollection` subclasses and `VectorStoreField` (section 11) |
| `ChatCompletionAgent(service_id=..., kernel=...)` | `service_id` argument gone | `service=` or a `kernel=` holding the service |
| `agent.invoke(chat_history)` | Changed in the 1.2x agent GA | `get_response` / `invoke` with `messages=` and `thread=` |
| `AgentGroupChat` + strategies | Superseded | `GroupChatOrchestration` |
| `semantic_kernel.connectors.ai.azure_open_ai` | Never existed | `semantic_kernel.connectors.ai.open_ai.AzureChatCompletion` |
| `MCPClient`, `MCPServer`, `MCPToolPlugin` | Never existed | `MCPStdioPlugin` / `MCPStreamableHttpPlugin`, `kernel.as_mcp_server()` |

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | 1.45.0 | Rewritten. The previous edition documented planners removed in 1.33.0, a pre-1.34 vector store API, and several modules that never existed (`semantic_kernel.interop.a2a`, `semantic_kernel.connectors.a2a`, `MCPClient`/`MCPServer`, `connectors.ai.azure_open_ai`). It also claimed `pip install --pre` was required, which is not the case. All examples now use APIs verified against an installed copy; added orchestrations, the Process framework and the Agent Framework migration path. |
| 2026-06-04 | 1.43.0 | Version bump (unverified). |
| 2026-05-14 | 1.42.0 | Version bump; `mcp>=1.26.0` requirement noted. |
| 2026-04-08 | 1.41.2 | MCP, A2A and Agent Framework sections added (since corrected). |
| November 2025 | 1.38.0 | First documented version. |

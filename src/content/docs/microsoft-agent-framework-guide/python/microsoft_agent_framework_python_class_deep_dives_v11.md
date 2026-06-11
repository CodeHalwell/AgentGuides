---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 11"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.8.1: OpenAIChatClient+OpenAIChatOptions+ReasoningOptions, OpenAIChatCompletionClient+OpenAIChatCompletionOptions, FoundryChatClient+FoundrySettings+built-in tools, FoundryMemoryProvider, FoundryLocalClient+FoundryLocalSettings, AzureAISearchContextProvider, CosmosCheckpointStorage+CosmosHistoryProvider, OrchestrationState, ClaudeAgent+ClaudeAgentSettings, AgentFunctionApp."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 34
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 11

Verified against **agent-framework 1.8.1** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source at `/usr/local/lib/python3.11/dist-packages/`. No API name has been guessed or
inferred from documentation alone.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, `AgentMiddleware`, `ChatMiddleware`, `FunctionMiddleware`, `CompactionProvider`, `ToolResultCompactionStrategy`, `TokenBudgetComposedStrategy`, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — `BackgroundAgentsProvider`, `MemoryContextProvider`, `TodoProvider`, `AgentModeProvider`, `SummarizationStrategy`, `ContextWindowCompactionStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`, `WorkflowViz`, `MCPStreamableHTTPTool` + `MCPWebsocketTool`
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — `Message` + `Content`, `ChatOptions` + `ChatResponse`, `ResponseStream`, `AgentContext`, `FunctionalWorkflow` + `StepWrapper`, `WorkflowEvent` taxonomy, `SkillsSource` composition, `EvalItem` + `EvalResults`, `TokenizerProtocol`, `ConversationSplit`
- [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) — `Executor` + `@handler` + `@executor`, `AgentExecutor`, edge groups, `Runner`, `SessionContext`, `AgentSession`, `BaseChatClient`, `SecretString`, `WorkflowCheckpoint`, exception hierarchy
- [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) — `ExperimentalFeature`, `WorkflowRunState`, `WorkflowExecutor`, `AgentResponse`, `BaseEmbeddingClient`, `FunctionInvocationConfiguration`, `ClassSkill`, `Annotation`, capability protocols, middleware layers
- [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) — `ContextProvider`, `BackgroundTaskInfo`, `GroupChatBuilder`, `HandoffBuilder`, `MagenticBuilder`, `SequentialBuilder`, `ConcurrentBuilder`, `AgentFactory`, `WorkflowFactory`, `SecureAgentConfig`, `FunctionalWorkflowAgent`, `ObservabilitySettings`
- [Vol. 8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) — `AgentFileStore` hierarchy, `FileAccessProvider`, `MCPSkill` + `MCPSkillsSource`, `ToolMode`, `AgentEvalConverter` + `CheckResult` + `RubricScore`, `ChatContext`, `WorkflowAgent` + `WorkflowContext`, `TruncationStrategy`, `HistoryProvider` + `InMemoryHistoryProvider`, `DelegatingSkillsSource` + `InMemorySkillsSource` + `FunctionInvocationContext`
- [Vol. 9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) — `OllamaChatClient`, `PurviewPolicyMiddleware`, `DurableAIAgent` + `DurableAIAgentClient` + `DurableAIAgentWorker`, `GitHubCopilotAgent`, `HyperlightExecuteCodeTool` + `HyperlightCodeActProvider`, `Mem0ContextProvider`, `RedisContextProvider` + `RedisHistoryProvider`, `StandardMagenticManager` + `MagenticContext`, `FileSkillsSource` + `FilteringSkillsSource`
- [Vol. 10](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10/) — `A2AAgent` + `A2AAgentSession` + `A2AContinuationToken`, `A2AExecutor`, `AGUIChatClient` + `AGUIHttpService`, `AnthropicClient` + `ThinkingConfig`, `BedrockChatClient` + `BedrockGuardrailConfig`, `CopilotStudioAgent`, `GroupChatOrchestrator`, `AgentBasedGroupChatOrchestrator`, `HandoffBuilder` (orchestrations), `ExternalInputRequest` + `MCPToolApprovalRequest`

This volume covers **ten new class groups** spanning the OpenAI client family, three Azure
AI Foundry integration packages, cloud persistence backends, cross-framework protocol
support, and serverless hosting:

1. **OpenAI Responses API** — `OpenAIChatClient` with reasoning, background mode, and
   multi-turn conversation state
2. **OpenAI Chat Completions API** — `OpenAIChatCompletionClient` as a leaner alternative
3. **Azure AI Foundry** — `FoundryChatClient` with every built-in cloud tool wired to
   Bing, SharePoint, Fabric, Memory Search, Computer Use, and Browser Automation
4. **Foundry Memory** — `FoundryMemoryProvider` for Azure AI Foundry's semantic memory
   store
5. **Foundry Local** — `FoundryLocalClient` for running phi-4-mini and other models
   entirely on-device
6. **Azure AI Search** — `AzureAISearchContextProvider` in both semantic and new agentic
   retrieval modes
7. **Azure Cosmos DB** — `CosmosCheckpointStorage` + `CosmosHistoryProvider` for
   cloud-backed persistence
8. **OrchestrationState** — unified serialisable state shared by all built-in
   orchestrators
9. **Claude Agent** — `ClaudeAgent` wrapping the Claude Code CLI as an agent-framework
   participant
10. **Azure Functions hosting** — `AgentFunctionApp` for serverless durable entity
    deployment

---

## Table of Contents

1. [`OpenAIChatClient` + `OpenAIChatOptions` + `ReasoningOptions` + `StreamOptions` + `OpenAIContinuationToken`](#1-openaichatclient--openaichatoptions--reasoningoptions--streamoptions--openaicontinuationtoken)
2. [`OpenAIChatCompletionClient` + `OpenAIChatCompletionOptions`](#2-openaichatcompletionclient--openaichatcompletionoptions)
3. [`FoundryChatClient` + `FoundrySettings` + built-in cloud tools](#3-foundrychatclient--foundrysettings--built-in-cloud-tools)
4. [`FoundryMemoryProvider` + `FoundryProjectSettings`](#4-foundrymemoryprovider--foundryprojectsettings)
5. [`FoundryLocalClient` + `FoundryLocalSettings` + `FoundryLocalChatOptions`](#5-foundrylocalclient--foundrylocalsettings--foundrylocalachatoptions)
6. [`AzureAISearchContextProvider` + `AzureAISearchSettings`](#6-azureaisearchcontextprovider--azureaisearchsettings)
7. [`CosmosCheckpointStorage` + `CosmosHistoryProvider`](#7-cosmoscheckpointstorage--cosmoshistoryprovider)
8. [`OrchestrationState`](#8-orchestrationstate)
9. [`ClaudeAgent` + `ClaudeAgentSettings` + `ClaudeAgentOptions`](#9-claudeagent--claudeagentsettings--claudeagentoptions)
10. [`AgentFunctionApp`](#10-agentfunctionapp)

---

## 1. `OpenAIChatClient` + `OpenAIChatOptions` + `ReasoningOptions` + `StreamOptions` + `OpenAIContinuationToken`

**Source:** `agent_framework_openai._chat_client`
**Package:** `pip install agent-framework` (bundled with core)

`OpenAIChatClient` is the framework's primary OpenAI integration and the one you reach for
most often. It wraps the **Responses API** (the newer, stateful, multi-turn OpenAI
surface), not the Chat Completions API — see section 2 for the latter. It implements
`FunctionInvocationLayer`, `ChatMiddlewareLayer`, and `ChatTelemetryLayer`, giving you
tool-calling, middleware chaining, and OpenTelemetry tracing with no extra wiring.

The client has two construction overloads: one for **direct OpenAI** (API key) and one
for **Azure OpenAI** (endpoint + credential). Both share the same runtime interface.

### Constructor signatures

```python
class OpenAIChatClient(
    FunctionInvocationLayer[OpenAIChatOptionsT],
    ChatMiddlewareLayer[OpenAIChatOptionsT],
    ChatTelemetryLayer[OpenAIChatOptionsT],
    RawOpenAIChatClient[OpenAIChatOptionsT],
):
    # Overload 1 — OpenAI direct
    def __init__(
        self,
        model: str | None = None,           # OPENAI_CHAT_MODEL / OPENAI_MODEL
        *,
        api_key: str | Callable[[], str | Awaitable[str]] | None = None,
        org_id: str | None = None,          # OPENAI_ORG_ID
        base_url: str | None = None,        # OPENAI_BASE_URL
        default_headers: Mapping[str, str] | None = None,
        async_client: AsyncOpenAI | None = None,
        instruction_role: str | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        additional_properties: dict[str, Any] | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...

    # Overload 2 — Azure OpenAI
    def __init__(
        self,
        model: str | None = None,           # AZURE_OPENAI_CHAT_MODEL / AZURE_OPENAI_MODEL
        *,
        azure_endpoint: str | None = None,  # AZURE_OPENAI_ENDPOINT
        credential: AzureCredentialTypes | AzureTokenProvider | None = None,
        api_version: str | None = None,
        # … same kwargs as overload 1 …
    ) -> None: ...
```

### `OpenAIChatOptions` TypedDict — all fields

```python
class OpenAIChatOptions(ChatOptions[ResponseFormatT], total=False):
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    stop: str | Sequence[str]
    seed: int
    logit_bias: dict[str | int, float]
    frequency_penalty: float
    presence_penalty: float
    tools: ToolTypes | Sequence[ToolTypes] | None
    tool_choice: ToolMode | Literal["auto", "required", "none"]
    allow_multiple_tool_calls: bool
    response_format: type[BaseModel] | Mapping[str, Any] | None
    metadata: dict[str, Any]
    user: str
    store: bool                         # store response for fine-tuning
    conversation_id: str                # multi-turn; auto-set by client
    instructions: str                   # per-request system prompt override
    include: list[str]                  # extra fields in the response
    max_tool_calls: int
    prompt: dict[str, Any]              # named prompt reference
    prompt_cache_key: str
    prompt_cache_retention: Literal["24h"]
    reasoning: ReasoningOptions         # o-model reasoning control
    verbosity: Literal["low", "medium", "high"]
    safety_identifier: str
    service_tier: Literal["auto", "default", "flex", "priority"]
    stream_options: StreamOptions
    top_logprobs: int
    truncation: Literal["auto", "disabled"]
    background: bool                    # true → fire-and-forget background run
    continuation_token: OpenAIContinuationToken
```

### `ReasoningOptions` and `StreamOptions`

```python
class ReasoningOptions(TypedDict, total=False):
    effort: Literal["none", "low", "medium", "high", "xhigh"]
    summary: Literal["auto", "concise", "detailed"]

class StreamOptions(TypedDict, total=False):
    include_usage: bool    # emit a usage event at end of stream
```

### `OpenAIContinuationToken`

`OpenAIContinuationToken` extends `ContinuationToken` and is automatically set in
`AgentResponse.continuation_token` when the Responses API returns a `previous_response_id`
(background operations, multi-turn state). Pass it back via
`options["continuation_token"]` to resume.

### Example 1 — basic agent with Azure OpenAI

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

async def main():
    client = OpenAIChatClient(
        azure_endpoint="https://my-hub.openai.azure.com/",
        api_version="2025-01-01-preview",
    )
    agent = client.as_agent(
        name="Helper",
        instructions="You are a concise assistant.",
    )
    response = await agent.run("Summarise the OSI model in one sentence.")
    print(response.text)

asyncio.run(main())
```

### Example 2 — reasoning (o3) with effort and summary

```python
from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions

client = OpenAIChatClient("o3")

options: OpenAIChatOptions = {
    "reasoning": {"effort": "high", "summary": "concise"},
    "max_tokens": 4096,
}

agent = client.as_agent(
    name="Reasoner",
    instructions="Think carefully before answering.",
)

async def run():
    response = await agent.run(
        "What is the optimal TSP tour for 5 cities?",
        options=options,
    )
    print(response.text)
```

### Example 3 — background mode (fire-and-forget) with continuation polling

```python
import asyncio
from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions

client = OpenAIChatClient("gpt-4.1")

async def run():
    agent = client.as_agent(name="BgAgent", instructions="You are helpful.")

    # Fire a long-running background task
    options: OpenAIChatOptions = {"background": True}
    response = await agent.run("Write a 2 000-word essay on climate tech.", options=options)

    token = response.continuation_token        # OpenAIContinuationToken
    assert token is not None

    # Poll until complete
    while True:
        result = await agent.run(
            [],                                # no new input — resume only
            options={"continuation_token": token},
        )
        if result.continuation_token is None:
            print(result.text)
            break
        token = result.continuation_token
        await asyncio.sleep(5)

asyncio.run(run())
```

### Example 4 — multi-turn conversation state via `conversation_id`

```python
from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions
from agent_framework import AgentSession

client = OpenAIChatClient()
agent = client.as_agent(name="Chat", instructions="You are a helpful tutor.")

async def multi_turn():
    session = AgentSession()

    # First turn — conversation_id is auto-populated by the client
    r1 = await agent.run("Explain async/await in Python.", session=session)
    print("R1:", r1.text[:100])

    # Second turn — the same conversation context is continued
    r2 = await agent.run("Give me a code example.", session=session)
    print("R2:", r2.text[:100])
```

### Example 5 — streaming with usage reporting

```python
from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions

client = OpenAIChatClient()
agent = client.as_agent(name="StreamAgent", instructions="Be brief.")

async def stream():
    options: OpenAIChatOptions = {"stream_options": {"include_usage": True}}
    async with agent.run_stream("Explain quantum entanglement.", options=options) as stream:
        async for update in stream:
            print(update.text, end="", flush=True)
        final = await stream.get_final_response()
        print(f"\nTokens: {final.usage}")
```

---

## 2. `OpenAIChatCompletionClient` + `OpenAIChatCompletionOptions`

**Source:** `agent_framework_openai._chat_completion_client`
**Package:** `pip install agent-framework`

`OpenAIChatCompletionClient` uses the **Chat Completions API** (`/v1/chat/completions`),
the classic stateless OpenAI endpoint. Prefer this when you need:
- Provider compatibility with non-OpenAI deployments that implement the Chat Completions
  spec (many local inference servers, third-party proxies)
- `logprobs` / `top_logprobs` for log-probability analysis
- `prediction` for output pre-filling (speculative decoding)
- Lower-cost, no-state-overhead streaming

Constructor signature is identical to `OpenAIChatClient` (same dual overloads for OpenAI
vs Azure OpenAI). The key difference is the `options` TypedDict:

### `OpenAIChatCompletionOptions` TypedDict

```python
class OpenAIChatCompletionOptions(ChatOptions[ResponseModelT], total=False):
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    stop: str | Sequence[str]
    seed: int
    logit_bias: dict[str | int, float]
    frequency_penalty: float
    presence_penalty: float
    tools: ToolTypes | Sequence[ToolTypes] | None
    tool_choice: ToolMode | Literal["auto", "required", "none"]
    allow_multiple_tool_calls: bool
    response_format: type[BaseModel] | Mapping[str, Any] | None
    metadata: dict[str, Any]
    user: str
    store: bool
    conversation_id: str
    instructions: str
    logprobs: bool                    # emit log-probabilities in the response
    top_logprobs: int                 # how many top tokens to include (0–20)
    prediction: Prediction            # pre-filled output for speculative decoding
    verbosity: Literal["low", "medium", "high"]
```

`Prediction` TypedDict has fields `type: Literal["content"]` and
`content: str | list[dict[str, Any]]`.

### Example 1 — basic Chat Completions agent

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient

async def main():
    client = OpenAIChatCompletionClient("gpt-4o")
    agent = client.as_agent(
        name="CompletionAgent",
        instructions="You are a concise code assistant.",
    )
    response = await agent.run("Write a Python hello-world in 3 lines.")
    print(response.text)

asyncio.run(main())
```

### Example 2 — structured output with Pydantic

```python
from pydantic import BaseModel
from agent_framework.openai import OpenAIChatCompletionClient, OpenAIChatCompletionOptions

class FileSummary(BaseModel):
    filename: str
    purpose: str
    lines_of_code: int

client = OpenAIChatCompletionClient()
options: OpenAIChatCompletionOptions[FileSummary] = {"response_format": FileSummary}

agent = client.as_agent(name="Summariser", instructions="Extract file metadata.")

async def summarise(file_content: str):
    response = await agent.run(
        f"Summarise this file:\n{file_content}",
        options=options,
    )
    summary: FileSummary = response.structured_output
    print(summary.model_dump_json(indent=2))
```

### Example 3 — log-probability analysis

```python
from agent_framework.openai import OpenAIChatCompletionClient, OpenAIChatCompletionOptions

client = OpenAIChatCompletionClient("gpt-4o-mini")
options: OpenAIChatCompletionOptions = {"logprobs": True, "top_logprobs": 5}

agent = client.as_agent(name="LogprobAgent", instructions="Answer yes or no only.")

async def analyse():
    response = await agent.run(
        "Is Python a statically typed language?",
        options=options,
    )
    # Raw logprobs accessible via response.metadata["logprobs"]
    print(response.text)
    print("Metadata:", response.metadata.get("logprobs"))
```

### Example 4 — speculative decoding with `prediction`

```python
from agent_framework.openai import OpenAIChatCompletionClient, OpenAIChatCompletionOptions

client = OpenAIChatCompletionClient("gpt-4o")

# Pre-fill the first few tokens of the expected response to reduce TTFT
options: OpenAIChatCompletionOptions = {
    "prediction": {
        "type": "content",
        "content": "```python\nimport asyncio\n",
    }
}

agent = client.as_agent(name="CodeGen", instructions="Write Python only.")

async def generate():
    response = await agent.run("Write a simple event loop example.", options=options)
    print(response.text)
```

---

## 3. `FoundryChatClient` + `FoundrySettings` + built-in cloud tools

**Source:** `agent_framework_foundry._chat_client`
**Package:** `pip install agent-framework-foundry`

`FoundryChatClient` is the Azure AI Foundry–first chat client. It is built on top of
`RawFoundryChatClient` which extends `RawOpenAIChatClient`, inheriting all Responses API
semantics and adding a wide suite of cloud-native tools backed by the **Azure AI Projects
SDK** (`azure-ai-projects`). Using any of these tools requires `allow_preview=True`.

### Constructor signature

```python
class FoundryChatClient(
    FunctionInvocationLayer[FoundryChatOptionsT],
    ChatMiddlewareLayer[FoundryChatOptionsT],
    ChatTelemetryLayer[FoundryChatOptionsT],
    RawFoundryChatClient[FoundryChatOptionsT],
):
    def __init__(
        self,
        *,
        project_endpoint: str | None = None,  # FOUNDRY_PROJECT_ENDPOINT
        project_client: AIProjectClient | None = None,
        model: str | None = None,             # FOUNDRY_MODEL
        credential: AzureCredentialTypes | AzureTokenProvider | None = None,
        allow_preview: bool | None = None,    # required for preview tools
        default_headers: Mapping[str, str] | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
        instruction_role: str | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        additional_properties: dict[str, Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
    ) -> None: ...
```

### `FoundrySettings` TypedDict

```python
class FoundrySettings(TypedDict, total=False):
    model: str | None            # FOUNDRY_MODEL
    project_endpoint: str | None # FOUNDRY_PROJECT_ENDPOINT
```

### Built-in tool factory methods on `RawFoundryChatClient`

| Method | Tool type | Notes |
|--------|-----------|-------|
| `get_code_interpreter_tool()` | `CodeInterpreterTool` | File I/O, execution sandbox |
| `get_file_search_tool(vector_store_ids)` | `ProjectsFileSearchTool` | RAG over VS |
| `get_web_search_tool(...)` | `WebSearchTool` | Bing web search |
| `get_bing_grounding_tool(...)` | `BingGroundingTool` | Bing grounding with config |
| `get_bing_custom_search_tool(...)` | `BingCustomSearchPreviewTool` | Custom Bing instance |
| `get_image_generation_tool()` | `ImageGenTool` | DALL-E image generation |
| `get_mcp_tool(server_label, ...)` | `FoundryMCPTool` | MCP tool hosted in Foundry |
| `get_azure_ai_search_tool(...)` | `AzureAISearchTool` | Azure AI Search grounding |
| `get_sharepoint_tool(...)` | `SharepointPreviewTool` | SharePoint grounding |
| `get_fabric_tool(...)` | `MicrosoftFabricPreviewTool` | Microsoft Fabric |
| `get_memory_search_tool(...)` | `MemorySearchPreviewTool` | Foundry Memory Search |
| `get_computer_use_tool()` | `ComputerUsePreviewTool` | Computer use (preview) |
| `get_browser_automation_tool(...)` | `BrowserAutomationPreviewTool` | Browser automation (preview) |
| `get_a2a_tool(agent_id, ...)` | `A2APreviewTool` | Connected agent via A2A |

### Example 1 — basic agent with managed identity

```python
import asyncio
from azure.identity.aio import DefaultAzureCredential
from agent_framework.foundry import FoundryChatClient

async def main():
    async with DefaultAzureCredential() as credential:
        client = FoundryChatClient(
            project_endpoint="https://my-hub.api.azureml.ms",
            model="gpt-4.1",
            credential=credential,
        )
        agent = client.as_agent(
            name="FoundryAgent",
            instructions="You are a helpful enterprise assistant.",
        )
        response = await agent.run("What are the key features of Azure AI Foundry?")
        print(response.text)

asyncio.run(main())
```

### Example 2 — Bing grounding + Azure AI Search

```python
from azure.identity.aio import DefaultAzureCredential
from agent_framework.foundry import FoundryChatClient
from azure.ai.projects.models import BingGroundingSearchConfiguration

async def research_agent():
    async with DefaultAzureCredential() as credential:
        client = FoundryChatClient(
            project_endpoint="https://my-hub.api.azureml.ms",
            model="gpt-4.1",
            credential=credential,
            allow_preview=True,
        )

        # Bing grounding tool — live web search with citations
        bing_tool = client.get_bing_grounding_tool(
            connection_id="/subscriptions/.../connections/bing-conn",
            search_config=BingGroundingSearchConfiguration(count=10, market="en-US"),
        )

        # Azure AI Search for internal knowledge base
        search_tool = client.get_azure_ai_search_tool(
            index_connection_id="/subscriptions/.../connections/aisearch-conn",
            index_name="enterprise-kb",
        )

        agent = client.as_agent(
            name="ResearchAgent",
            instructions="Use the available tools to answer grounded in facts.",
            tools=[bing_tool, search_tool],
        )
        response = await agent.run("What are the latest Azure AI announcements?")
        print(response.text)
```

### Example 3 — SharePoint grounding for internal documents

```python
from azure.identity.aio import DefaultAzureCredential
from agent_framework.foundry import FoundryChatClient

async def sharepoint_agent():
    async with DefaultAzureCredential() as credential:
        client = FoundryChatClient(
            project_endpoint="https://my-hub.api.azureml.ms",
            model="gpt-4.1",
            credential=credential,
            allow_preview=True,
        )

        sharepoint_tool = client.get_sharepoint_tool(
            connection_id="/subscriptions/.../connections/sharepoint-conn",
            site_names=["EngineeringDocs", "PolicyPortal"],
        )

        agent = client.as_agent(
            name="DocsAgent",
            instructions="Answer questions about internal policies.",
            tools=[sharepoint_tool],
        )
        response = await agent.run("What is the expense approval policy for over $1000?")
        print(response.text)
```

### Example 4 — Computer Use agent (preview)

```python
from azure.identity.aio import DefaultAzureCredential
from agent_framework.foundry import FoundryChatClient

async def computer_use_agent():
    async with DefaultAzureCredential() as credential:
        client = FoundryChatClient(
            project_endpoint="https://my-hub.api.azureml.ms",
            model="computer-use-preview",
            credential=credential,
            allow_preview=True,
        )

        computer_tool = client.get_computer_use_tool()

        agent = client.as_agent(
            name="BrowserAgent",
            instructions="Use the computer tool to perform web tasks.",
            tools=[computer_tool],
        )
        response = await agent.run("Navigate to https://example.com and extract the page title.")
        print(response.text)
```

### Example 5 — Cross-agent A2A tool

```python
from azure.identity.aio import DefaultAzureCredential
from agent_framework.foundry import FoundryChatClient

async def connected_agent():
    async with DefaultAzureCredential() as credential:
        client = FoundryChatClient(
            project_endpoint="https://my-hub.api.azureml.ms",
            model="gpt-4.1",
            credential=credential,
            allow_preview=True,
        )

        # Connect to another Foundry-hosted agent via A2A tool
        data_agent_tool = client.get_a2a_tool(
            agent_id="asst_data_analyst_xyz",
            description="Runs data analysis queries and returns results as JSON.",
        )

        orchestrator = client.as_agent(
            name="Orchestrator",
            instructions="Delegate data queries to the data analyst.",
            tools=[data_agent_tool],
        )
        response = await orchestrator.run("What were our top 5 products last quarter?")
        print(response.text)
```

---

## 4. `FoundryMemoryProvider` + `FoundryProjectSettings`

**Source:** `agent_framework_foundry._memory_provider`
**Package:** `pip install agent-framework-foundry`

`FoundryMemoryProvider` integrates Azure AI Foundry's managed **Memory Store** — a
persistent semantic memory service — as a `ContextProvider` hook. On each agent run it
(a) fetches static profile memories on first invocation, (b) searches for contextual
memories relevant to the user's input, then (c) injects the combined result into the
session context before the LLM call. After the run it fires a debounced
`begin_update_memories` to store the new exchange.

### Constructor signature

```python
class FoundryMemoryProvider(ContextProvider):
    DEFAULT_SOURCE_ID: ClassVar[str] = "foundry_memory"
    DEFAULT_CONTEXT_PROMPT = "## Memories\nConsider the following memories when answering user questions:"

    def __init__(
        self,
        source_id: str = DEFAULT_SOURCE_ID,
        *,
        project_client: AIProjectClient | None = None,
        project_endpoint: str | None = None,  # FOUNDRY_PROJECT_ENDPOINT
        credential: AzureCredentialTypes | None = None,
        allow_preview: bool | None = None,
        memory_store_name: str,               # required — name of the memory store
        scope: str | None = None,             # logical namespace (e.g. user ID)
        context_prompt: str | None = None,    # preamble injected before memories
        update_delay: int = 300,              # seconds before writing new memories
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...
```

**Lifecycle hooks implemented:** `before_run` (inject memories) and `after_run`
(debounced write). The provider works only inside an `async with` block (the
`AIProjectClient` is async context-managed internally).

### Example 1 — personalised assistant with long-term memory

```python
import asyncio
from azure.identity.aio import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient, FoundryMemoryProvider

async def main():
    async with DefaultAzureCredential() as credential:
        client = FoundryChatClient(
            project_endpoint="https://my-hub.api.azureml.ms",
            credential=credential,
            allow_preview=True,
        )

        memory = FoundryMemoryProvider(
            project_endpoint="https://my-hub.api.azureml.ms",
            credential=credential,
            allow_preview=True,
            memory_store_name="user-memories",
            scope="user-42",             # partition memories per user
            update_delay=0,              # write immediately for demos
        )

        async with memory:
            agent = Agent(
                client=client,
                instructions="You are a personalised assistant. Use memories to personalise responses.",
                context_providers=[memory],
            )

            # Session 1
            await agent.run("My name is Alice and I prefer concise answers.")
            # Session 2 (later) — Alice's preferences are retrieved from memory
            response = await agent.run("What's the best way to learn Rust?")
            print(response.text)

asyncio.run(main())
```

### Example 2 — shared team memory with custom scope

```python
from azure.identity.aio import DefaultAzureCredential
from agent_framework.foundry import FoundryChatClient, FoundryMemoryProvider

async def team_agent(team_id: str):
    async with DefaultAzureCredential() as credential:
        client = FoundryChatClient(
            project_endpoint="https://my-hub.api.azureml.ms",
            credential=credential,
        )

        # Scope to a team rather than an individual user
        memory = FoundryMemoryProvider(
            project_endpoint="https://my-hub.api.azureml.ms",
            credential=credential,
            memory_store_name="team-memories",
            scope=f"team-{team_id}",
            context_prompt="## Team Knowledge\nApply the following team context:",
            update_delay=60,   # 1-minute debounce
        )

        async with memory:
            agent = client.as_agent(
                name="TeamAssistant",
                instructions="You are a team knowledge assistant.",
                context_providers=[memory],
            )
            response = await agent.run("Summarise our current sprint goals.")
            print(response.text)
```

### Example 3 — incremental memory updates across multiple turns

```python
from agent_framework import Agent, AgentSession
from agent_framework.foundry import FoundryChatClient, FoundryMemoryProvider
from azure.identity.aio import DefaultAzureCredential

async def multi_turn_memory():
    async with DefaultAzureCredential() as credential:
        client = FoundryChatClient(
            project_endpoint="https://my-hub.api.azureml.ms",
            credential=credential,
        )
        memory = FoundryMemoryProvider(
            project_endpoint="https://my-hub.api.azureml.ms",
            credential=credential,
            memory_store_name="conversations",
            scope="user-99",
        )
        async with memory:
            agent = Agent(
                client=client,
                instructions="Maintain continuity across our conversation.",
                context_providers=[memory],
            )
            session = AgentSession()

            for question in [
                "I work in healthcare IT.",
                "Our main challenge is HIPAA compliance.",
                "What encryption standards should we use?",
            ]:
                response = await agent.run(question, session=session)
                print(f"Q: {question}\nA: {response.text[:200]}\n")
```

---

## 5. `FoundryLocalClient` + `FoundryLocalSettings` + `FoundryLocalChatOptions`

**Source:** `agent_framework_foundry_local._foundry_local_client`
**Package:** `pip install agent-framework-foundry-local`

`FoundryLocalClient` runs **small language models entirely on the local machine** via the
[Foundry Local](https://github.com/Azure/azure-ai-foundry-model-inference) runtime. It
bootstraps the Foundry Local service, downloads the requested model into the local cache,
and exposes an OpenAI-compatible endpoint — all with the standard
`FunctionInvocationLayer`/`ChatMiddlewareLayer`/`ChatTelemetryLayer` stack.

It inherits from `RawOpenAIChatCompletionClient` (Chat Completions, not Responses API),
so `OpenAIChatCompletionOptions` semantics apply for generation parameters.

### Constructor signature

```python
class FoundryLocalClient(
    FunctionInvocationLayer[FoundryLocalChatOptionsT],
    ChatMiddlewareLayer[FoundryLocalChatOptionsT],
    ChatTelemetryLayer[FoundryLocalChatOptionsT],
    RawOpenAIChatCompletionClient[FoundryLocalChatOptionsT],
):
    def __init__(
        self,
        model: str | None = None,           # FOUNDRY_LOCAL_MODEL
        *,
        bootstrap: bool = True,             # start service if not running
        timeout: float | None = None,       # request timeout (seconds)
        prepare_model: bool = True,         # download + load on __init__
        device: DeviceType | None = None,   # CPU, GPU, Auto (from foundry_local.models)
        additional_properties: dict[str, Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str = "utf-8",
    ) -> None: ...
```

### `FoundryLocalChatOptions`

Inherits all `OpenAIChatCompletionOptions` fields. Options not supported for local
inference (`user`, `store`) are typed as `None`. Adds:

```python
class FoundryLocalChatOptions(ChatOptions[ResponseModelT], total=False):
    extra_body: dict[str, Any]    # model-specific params not in standard API
```

### Example 1 — first run with phi-4-mini

```python
import asyncio
from agent_framework.foundry import FoundryLocalClient

async def main():
    # Downloads phi-4-mini (~2.7 GB) and starts the local inference service
    client = FoundryLocalClient("phi-4-mini")

    agent = client.as_agent(
        name="LocalAgent",
        instructions="You are a helpful local assistant.",
    )
    response = await agent.run("Explain gradient descent in two sentences.")
    print(response.text)

asyncio.run(main())
```

### Example 2 — deferred model loading for faster startup

```python
from agent_framework.foundry import FoundryLocalClient

# prepare_model=False → skip download/load at init; the model loads on first request
client = FoundryLocalClient("phi-4-mini", prepare_model=False)

# Manually control loading when ready
manager = client.manager
manager.download_model("phi-4-mini")
manager.load_model("phi-4-mini")

agent = client.as_agent(name="OfflineAgent", instructions="Assist with coding.")

async def run():
    return await agent.run("Write a Python singleton class.")
```

### Example 3 — GPU inference with device selection

```python
from foundry_local.models import DeviceType
from agent_framework.foundry import FoundryLocalClient

# Force GPU execution (requires compatible hardware + model variant)
client = FoundryLocalClient("phi-4-mini", device=DeviceType.GPU)

agent = client.as_agent(name="GPUAgent", instructions="You are a fast local assistant.")

async def run():
    response = await agent.run("Describe CUDA memory management.")
    print(response.text)
```

### Example 4 — list available local models

```python
from agent_framework.foundry import FoundryLocalClient

# bootstrap=False → don't start service, just inspect catalog
client = FoundryLocalClient("phi-4-mini", bootstrap=False, prepare_model=False)

for model in client.manager.list_catalog_models():
    print(f"- {model.alias:<30} task={model.task}  id={model.id}")
```

### Example 5 — with tool calling

```python
import asyncio
from agent_framework import tool
from agent_framework.foundry import FoundryLocalClient

@tool
def get_system_time() -> str:
    """Return the current system time."""
    from datetime import datetime
    return datetime.now().isoformat()

async def main():
    client = FoundryLocalClient("phi-4-mini")
    agent = client.as_agent(
        name="ClockAgent",
        instructions="You can check the system time when asked.",
        tools=[get_system_time],
    )
    response = await agent.run("What time is it right now?")
    print(response.text)

asyncio.run(main())
```

---

## 6. `AzureAISearchContextProvider` + `AzureAISearchSettings`

**Source:** `agent_framework_azure_ai_search._context_provider`
**Package:** `pip install agent-framework-azure-ai-search`

`AzureAISearchContextProvider` is a `ContextProvider` that performs a search in Azure AI
Search before every agent invocation and injects the results as context messages. It
supports two modes:

| Mode | How it works | When to use |
|------|-------------|-------------|
| `"semantic"` | Keyword + vector + semantic reranking against a plain index | Full control over index schema |
| `"agentic"` | Multi-step reasoning retrieval via a Knowledge Base | Complex RAG where the retriever reasons over multiple sub-queries |

### Constructor signature

```python
class AzureAISearchContextProvider(ContextProvider):
    DEFAULT_SOURCE_ID: ClassVar[str] = "azure_ai_search"

    def __init__(
        self,
        source_id: str = DEFAULT_SOURCE_ID,
        endpoint: str | None = None,           # AZURE_SEARCH_ENDPOINT
        index_name: str | None = None,         # AZURE_SEARCH_INDEX_NAME
        api_key: str | AzureKeyCredential | None = None,
        credential: AzureCredentialTypes | None = None,
        *,
        mode: Literal["semantic", "agentic"] = "semantic",
        top_k: int = 5,                        # number of results to retrieve
        semantic_configuration_name: str | None = None,
        vector_field_name: str | None = None,  # field for vector queries
        embedding_function: EmbeddingFunction | None = None,
        context_prompt: str | None = None,     # preamble before results
        # agentic-mode only
        azure_openai_resource_url: str | None = None,
        model: str | None = None,
        knowledge_base_name: str | None = None,
        retrieval_instructions: str | None = None,
        azure_openai_api_key: str | None = None,
        knowledge_base_output_mode: Literal["extractive_data", "answer_synthesis"] = "extractive_data",
        retrieval_reasoning_effort: Literal["minimal", "medium", "low"] = "minimal",
        agentic_message_history_count: int = 10,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...
```

### `AzureAISearchSettings` TypedDict

```python
class AzureAISearchSettings(TypedDict, total=False):
    endpoint: str | None          # AZURE_SEARCH_ENDPOINT
    index_name: str | None        # AZURE_SEARCH_INDEX_NAME
    knowledge_base_name: str | None  # AZURE_SEARCH_KNOWLEDGE_BASE_NAME
    api_key: SecretString | None  # AZURE_SEARCH_API_KEY
```

### Example 1 — semantic mode with managed identity

```python
import asyncio
from azure.identity.aio import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework_azure_ai_search import AzureAISearchContextProvider

async def main():
    async with DefaultAzureCredential() as credential:
        search_provider = AzureAISearchContextProvider(
            endpoint="https://my-search.search.windows.net",
            index_name="product-docs",
            credential=credential,
            mode="semantic",
            semantic_configuration_name="default",
            top_k=5,
        )

        client = OpenAIChatClient()
        agent = Agent(
            client=client,
            instructions="Answer questions about our products using the provided context.",
            context_providers=[search_provider],
        )

        response = await agent.run("What are the key features of the Pro plan?")
        print(response.text)

asyncio.run(main())
```

### Example 2 — vector search with a custom embedding function

```python
from azure.identity.aio import DefaultAzureCredential
from agent_framework_azure_ai_search import AzureAISearchContextProvider
from agent_framework.foundry import FoundryChatClient
from openai import AsyncOpenAI

async def get_embedding(text: str) -> list[float]:
    """Compute text embedding via OpenAI."""
    openai = AsyncOpenAI()
    result = await openai.embeddings.create(input=text, model="text-embedding-3-small")
    return result.data[0].embedding

async def vector_agent():
    async with DefaultAzureCredential() as credential:
        search_provider = AzureAISearchContextProvider(
            endpoint="https://my-search.search.windows.net",
            index_name="vector-docs",
            credential=credential,
            mode="semantic",
            vector_field_name="content_vector",
            embedding_function=get_embedding,
            top_k=8,
            context_prompt="Use the following context chunks to answer:",
        )

        client = FoundryChatClient(
            project_endpoint="https://my-hub.api.azureml.ms",
            credential=credential,
        )
        agent = client.as_agent(
            name="RAGAgent",
            instructions="Provide grounded answers from the retrieved context.",
            context_providers=[search_provider],
        )
        response = await agent.run("How does our caching strategy handle cache invalidation?")
        print(response.text)
```

### Example 3 — agentic retrieval mode with Knowledge Base

```python
from azure.identity.aio import DefaultAzureCredential
from agent_framework_azure_ai_search import AzureAISearchContextProvider
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent

async def agentic_rag():
    async with DefaultAzureCredential() as credential:
        # Agentic mode: the retriever reasons over the question before searching
        search_provider = AzureAISearchContextProvider(
            endpoint="https://my-search.search.windows.net",
            credential=credential,
            mode="agentic",
            knowledge_base_name="enterprise-kb",
            azure_openai_resource_url="https://my-aoai.openai.azure.com/",
            model="gpt-4o",
            retrieval_instructions="Focus on compliance-related content.",
            retrieval_reasoning_effort="medium",
            knowledge_base_output_mode="answer_synthesis",
            agentic_message_history_count=5,
        )

        agent = Agent(
            client=OpenAIChatClient(),
            instructions="Answer complex enterprise questions using retrieved knowledge.",
            context_providers=[search_provider],
        )
        response = await agent.run(
            "Compare our GDPR compliance posture with SOC 2 Type II requirements."
        )
        print(response.text)
```

### Example 4 — combining AI Search with Foundry Memory

```python
from azure.identity.aio import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.foundry import FoundryMemoryProvider
from agent_framework_azure_ai_search import AzureAISearchContextProvider

async def rich_context_agent():
    async with DefaultAzureCredential() as credential:
        memory = FoundryMemoryProvider(
            project_endpoint="https://my-hub.api.azureml.ms",
            credential=credential,
            memory_store_name="user-prefs",
            scope="user-7",
        )
        search_ctx = AzureAISearchContextProvider(
            endpoint="https://my-search.search.windows.net",
            credential=credential,
            mode="semantic",
            index_name="company-handbook",
            top_k=3,
        )

        async with memory:
            agent = Agent(
                client=OpenAIChatClient(),
                instructions="Use memory and policy documents to give personalised guidance.",
                context_providers=[memory, search_ctx],
            )
            response = await agent.run("What is our parental leave policy?")
            print(response.text)
```

---

## 7. `CosmosCheckpointStorage` + `CosmosHistoryProvider`

**Source:** `agent_framework_azure_cosmos._checkpoint_storage`, `._history_provider`
**Package:** `pip install agent-framework-azure-cosmos`

These two classes provide **Azure Cosmos DB NoSQL** as a persistent backend for workflow
checkpoints and conversation history respectively.

`CosmosCheckpointStorage` implements the `CheckpointStorage` protocol and stores
`WorkflowCheckpoint` documents with partition key `/workflow_name`. It uses the same
hybrid JSON + pickle encoding as `FileCheckpointStorage`, so complex Python objects
round-trip faithfully.

`CosmosHistoryProvider` implements `HistoryProvider` and persists conversation messages
(inputs, outputs, optionally context) keyed by `session_id`. It exposes granular
`store_inputs`, `store_outputs`, `store_context_messages` switches.

### Constructor signatures

```python
class CosmosCheckpointStorage:
    def __init__(
        self,
        *,
        endpoint: str | None = None,          # AZURE_COSMOS_ENDPOINT
        database_name: str | None = None,     # AZURE_COSMOS_DATABASE_NAME
        container_name: str | None = None,    # AZURE_COSMOS_CONTAINER_NAME
        credential: str | AzureCredentialTypes | None = None,  # key or RBAC
        cosmos_client: CosmosClient | None = None,             # pre-built client
        container_client: ContainerProxy | None = None,        # pre-built proxy
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
        allowed_checkpoint_types: list[str] | None = None,  # "module:qualname"
    ) -> None: ...


class CosmosHistoryProvider(HistoryProvider):
    DEFAULT_SOURCE_ID: ClassVar[str] = "azure_cosmos_history"

    def __init__(
        self,
        source_id: str = DEFAULT_SOURCE_ID,
        *,
        load_messages: bool = True,           # load history before each run
        store_outputs: bool = True,           # persist assistant messages
        store_inputs: bool = True,            # persist user messages
        store_context_messages: bool = False, # persist provider context msgs
        store_context_from: set[str] | None = None,  # whitelist by source_id
        endpoint: str | None = None,
        database_name: str | None = None,
        container_name: str | None = None,
        credential: str | AzureCredentialTypes | None = None,
        cosmos_client: CosmosClient | None = None,
        container_client: ContainerProxy | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...
```

### Example 1 — Cosmos-backed workflow checkpoint

```python
import asyncio
from azure.identity.aio import DefaultAzureCredential
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient
from agent_framework_azure_cosmos import CosmosCheckpointStorage

async def main():
    async with DefaultAzureCredential() as credential:
        storage = CosmosCheckpointStorage(
            endpoint="https://my-account.documents.azure.com:443/",
            credential=credential,
            database_name="agent-db",
            container_name="checkpoints",
        )

        # WorkflowBuilder with cloud-backed checkpoint storage
        # (assumes you have start_executor and other executors defined)
        # workflow = WorkflowBuilder(
        #     start_executor=start,
        #     checkpoint_storage=storage,
        # ).build()
        print("CosmosCheckpointStorage ready:", storage)

asyncio.run(main())
```

### Example 2 — Cosmos with account key and custom safe types

```python
from agent_framework_azure_cosmos import CosmosCheckpointStorage

# Using account key (for development / testing)
storage = CosmosCheckpointStorage(
    endpoint="https://my-account.documents.azure.com:443/",
    credential="primary-account-key-here",
    database_name="agent-db",
    container_name="workflow-checkpoints",
    # Allow additional application types in pickle deserialization
    allowed_checkpoint_types=[
        "my_app.state:ConversationState",
        "my_app.models:UserPreferences",
    ],
)
```

### Example 3 — Cosmos history provider for persistent conversations

```python
import asyncio
from azure.identity.aio import DefaultAzureCredential
from agent_framework import Agent, AgentSession
from agent_framework.openai import OpenAIChatClient
from agent_framework_azure_cosmos import CosmosHistoryProvider

async def main():
    async with DefaultAzureCredential() as credential:
        history = CosmosHistoryProvider(
            endpoint="https://my-account.documents.azure.com:443/",
            credential=credential,
            database_name="agent-db",
            container_name="conversations",
            load_messages=True,   # load previous context on each run
            store_outputs=True,   # persist every assistant reply
            store_inputs=True,    # persist every user message
        )

        client = OpenAIChatClient()
        agent = Agent(
            client=client,
            instructions="You are a helpful assistant. Refer to our previous conversations.",
            context_providers=[history],
        )

        # Session persists across process restarts; same session_id resumes context
        session = AgentSession()
        response = await agent.run("What did we discuss in our last session?", session=session)
        print(response.text)

asyncio.run(main())
```

### Example 4 — selective context storage (store only AI Search context, not memory)

```python
from azure.identity.aio import DefaultAzureCredential
from agent_framework_azure_cosmos import CosmosHistoryProvider
from agent_framework_azure_ai_search import AzureAISearchContextProvider

async def selective_storage():
    async with DefaultAzureCredential() as credential:
        search_provider = AzureAISearchContextProvider(
            endpoint="https://my-search.search.windows.net",
            credential=credential,
            index_name="docs",
        )
        history = CosmosHistoryProvider(
            endpoint="https://my-account.documents.azure.com:443/",
            credential=credential,
            database_name="agent-db",
            container_name="messages",
            store_context_messages=True,
            # Only persist context from AI Search, not from other providers
            store_context_from={"azure_ai_search"},
        )
        print("Providers configured:", search_provider.source_id, history.source_id)
```

### Example 5 — pre-built CosmosClient for connection pooling

```python
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential
from agent_framework_azure_cosmos import CosmosCheckpointStorage, CosmosHistoryProvider

async def shared_client():
    async with DefaultAzureCredential() as credential:
        # Share one CosmosClient across both storage backends
        cosmos = CosmosClient(
            url="https://my-account.documents.azure.com:443/",
            credential=credential,
        )

        checkpoint_storage = CosmosCheckpointStorage(
            cosmos_client=cosmos,
            database_name="agent-db",
            container_name="checkpoints",
        )
        history_provider = CosmosHistoryProvider(
            cosmos_client=cosmos,
            database_name="agent-db",
            container_name="history",
        )
        print("Shared client configured:", checkpoint_storage, history_provider)
```

---

## 8. `OrchestrationState`

**Source:** `agent_framework_orchestrations._orchestration_state`
**Package:** `pip install agent-framework-orchestrations`

`OrchestrationState` is a `@dataclass` that serves as the **unified serialisable state
container** shared by `GroupChatOrchestrator`, `AgentBasedGroupChatOrchestrator`,
`HandoffOrchestrator`, and the Magentic-One orchestration pattern. Each pattern stores its
pattern-specific data in the extensible `metadata` dict; the common fields cover all
orchestration concerns.

### Class definition

```python
@dataclass
class OrchestrationState:
    conversation: list[Message] = field(default_factory=list)
    round_index: int = 0
    orchestrator_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    task: Message | None = None

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrchestrationState: ...
```

| Field | Purpose |
|-------|---------|
| `conversation` | Full message history across all participants |
| `round_index` | Number of coordination rounds completed |
| `orchestrator_name` | Identifies which orchestrator owns this state |
| `metadata` | Pattern-specific extensions (e.g. Magentic-One task/progress ledger) |
| `task` | The primary task/question being orchestrated |

### Example 1 — reading GroupChat state after each round

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import GroupChatOrchestrator

async def main():
    client = OpenAIChatClient()

    writer = client.as_agent(name="Writer", instructions="Write persuasive content.")
    editor = client.as_agent(name="Editor", instructions="Review and improve writing.")

    def turn_selector(agents, state):
        # Alternating selection: Writer → Editor → Writer → …
        return agents[state.round_index % len(agents)]

    orchestrator = GroupChatOrchestrator(
        agents=[writer, editor],
        selection_function=turn_selector,
        termination_condition=lambda msgs: len(msgs) >= 4,
    )

    response = await orchestrator.run("Write a product launch announcement.")
    print(response.text)
    # Check the final round count via the orchestrator's last state
    # (state accessible through checkpoint storage when persistence is configured)

asyncio.run(main())
```

### Example 2 — custom metadata in OrchestrationState for topic tracking

```python
from agent_framework_orchestrations._orchestration_state import OrchestrationState
from agent_framework._types import Message

# Build state with custom metadata
state = OrchestrationState(
    task=Message(role="user", contents=["Analyse this dataset."]),
    orchestrator_name="DataAnalysisOrchestrator",
    metadata={
        "topics_covered": ["data_loading", "visualisation"],
        "confidence_scores": {"data_loading": 0.9, "visualisation": 0.85},
    },
)

# Round-trip serialisation (used internally by checkpoint storage)
serialised = state.to_dict()
restored = OrchestrationState.from_dict(serialised)
assert restored.metadata["topics_covered"] == ["data_loading", "visualisation"]
assert restored.task is not None
print("State round-trip OK:", restored.round_index, restored.orchestrator_name)
```

### Example 3 — inspecting state in a custom GroupChat selection function

```python
from agent_framework_orchestrations import GroupChatOrchestrator
from agent_framework_orchestrations._orchestration_state import OrchestrationState
from agent_framework._types import Message

def smart_selector(agents, state: OrchestrationState) -> "Agent":
    """Select the next speaker based on conversation content."""
    # Inspect who spoke last
    assistant_msgs = [m for m in state.conversation if m.role == "assistant"]
    if not assistant_msgs:
        return agents[0]

    last_speaker_name = assistant_msgs[-1].metadata.get("agent_name", "")
    # Avoid the same agent speaking twice in a row
    candidates = [a for a in agents if a.name != last_speaker_name]
    return candidates[0] if candidates else agents[0]
```

---

## 9. `ClaudeAgent` + `ClaudeAgentSettings` + `ClaudeAgentOptions`

**Source:** `agent_framework_claude._agent`
**Package:** `pip install agent-framework-claude`

`ClaudeAgent` wraps the **Claude Code CLI** (the `claude-agent-sdk`) as a first-class
agent-framework participant. Under the hood it converts `agent_framework.FunctionTool`
instances into MCP tools, hosts them through an in-process MCP server, and delegates
execution to the Claude SDK client. This means you can compose `ClaudeAgent` with any
other agent-framework orchestration pattern — Sequential, Concurrent, Handoff, GroupChat —
without friction.

`ClaudeAgent` extends `Agent` and adds `ClaudeAgentOptions` for Claude-specific controls:
model, permission mode, max turns, budget, and MCP server configuration.

### Constructor and settings

```python
class ClaudeAgentSettings(TypedDict, total=False):
    cli_path: str | None          # path to claude CLI binary; defaults to PATH lookup
    model: str | None             # model ID (e.g. "claude-opus-4-8")
    cwd: str | None               # working directory for the CLI
    permission_mode: str | None   # e.g. "acceptEdits", "bypassPermissions"
    max_turns: int | None         # maximum agentic turns
    max_budget_usd: float | None  # cost limit per run
```

`ClaudeAgentOptions` (TypedDict) extends `AgentRunOptions` and adds all SDK options
including `mcp_servers`, `tools`, `system_prompt`, `allowed_tools`, `disallowed_tools`,
`sandbox_settings`, and more (forward-referenced from `claude_agent_sdk`).

### Example 1 — ClaudeAgent with file system tools

```python
import asyncio
from agent_framework.claude import ClaudeAgent

async def main():
    agent = ClaudeAgent(
        name="FileAgent",
        instructions="You are a helpful coding assistant with file access.",
        # ClaudeAgent automatically gets Read, Write, Edit tools from the CLI
    )
    response = await agent.run("List the Python files in the current directory.")
    print(response.text)

asyncio.run(main())
```

### Example 2 — ClaudeAgent with custom FunctionTools exposed as MCP

```python
import asyncio
from agent_framework import tool
from agent_framework.claude import ClaudeAgent

@tool
def get_database_schema(table_name: str) -> str:
    """Return the schema for a database table."""
    schemas = {
        "users": "id INT, name VARCHAR(255), email VARCHAR(255), created_at TIMESTAMP",
        "orders": "id INT, user_id INT, amount DECIMAL, status VARCHAR(50)",
    }
    return schemas.get(table_name, f"Table {table_name!r} not found")

async def main():
    # FunctionTools are automatically wrapped as MCP tools for Claude
    agent = ClaudeAgent(
        name="DBAgent",
        instructions="Help with database queries. Use get_database_schema to check schemas.",
        tools=[get_database_schema],
    )
    response = await agent.run("What fields are available in the orders table?")
    print(response.text)

asyncio.run(main())
```

### Example 3 — ClaudeAgent in a HandoffOrchestration

```python
import asyncio
from agent_framework.claude import ClaudeAgent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import HandoffBuilder

async def main():
    # Triage agent (OpenAI) routes to specialised ClaudeAgent
    triage = OpenAIChatClient().as_agent(
        name="Triage",
        instructions="Route coding questions to CodingExpert.",
    )
    coding_expert = ClaudeAgent(
        name="CodingExpert",
        instructions="You are an expert software engineer. Solve coding problems precisely.",
    )

    workflow = (
        HandoffBuilder(start=triage)
        .add_target(coding_expert)
        .build()
    )

    response = await workflow.run("Fix the off-by-one error in my binary search.")
    print(response.text)

asyncio.run(main())
```

### Example 4 — model and budget controls

```python
import asyncio
from agent_framework.claude import ClaudeAgent

async def main():
    agent = ClaudeAgent(
        name="BudgetedAgent",
        instructions="Complete the task efficiently.",
        settings={
            "model": "claude-haiku-4-5-20251001",
            "max_turns": 10,
            "max_budget_usd": 0.05,
            "permission_mode": "acceptEdits",
        },
    )
    response = await agent.run("Refactor this Python function to use list comprehensions.")
    print(response.text)

asyncio.run(main())
```

### Example 5 — ClaudeAgent in a ConcurrentBuilder for parallel code review

```python
import asyncio
from agent_framework.claude import ClaudeAgent
from agent_framework_orchestrations import ConcurrentBuilder

async def parallel_review(code: str):
    security_reviewer = ClaudeAgent(
        name="SecurityReviewer",
        instructions="Review code for security vulnerabilities (OWASP Top 10, injection, etc.).",
    )
    performance_reviewer = ClaudeAgent(
        name="PerformanceReviewer",
        instructions="Review code for performance issues and optimisation opportunities.",
    )

    workflow = ConcurrentBuilder(
        participants=[security_reviewer, performance_reviewer],
    ).build()

    response = await workflow.run(f"Review this code:\n```python\n{code}\n```")
    for msg in response.messages:
        print(f"\n--- {msg.metadata.get('agent_name', 'Agent')} ---")
        print(msg.text)

asyncio.run(parallel_review("def get_user(id):\n    return db.execute(f'SELECT * FROM users WHERE id={id}')"))
```

---

## 10. `AgentFunctionApp`

**Source:** `agent_framework_azurefunctions._app`
**Package:** `pip install agent-framework-azurefunctions`

`AgentFunctionApp` extends `azure.durable_functions.DFApp` to host agent-framework agents
and workflows as **Azure Functions using the Durable Entities pattern**. Each registered
agent gets:

- A **Durable Entity** managing per-conversation state (history, session, continuation
  tokens)
- An **HTTP trigger** endpoint for REST API access (optional)
- An optional **MCP tool trigger** for tool-level invocations

The Durable Entities pattern gives you stateful, scalable agent hosting with built-in
replay, fan-out/fan-in, and external event integration — without managing VMs or
containers.

### Constructor signature

```python
class AgentFunctionApp(DFApp):
    def __init__(
        self,
        agents: list[SupportsAgentRun] | None = None,
        workflow: Workflow | None = None,
        http_auth_level: func.AuthLevel = func.AuthLevel.FUNCTION,
        enable_health_check: bool = True,
        enable_http_endpoints: bool = True,
        max_poll_retries: int = DEFAULT_MAX_POLL_RETRIES,     # 30
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,  # 1.0
        enable_mcp_tool_trigger: bool = False,
        default_callback: AgentResponseCallbackProtocol | None = None,
    ): ...
```

### Key methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `add_agent(agent, ...)` | `add_agent(agent, http_endpoint_enabled=True, mcp_tool_enabled=False)` | Register a new agent |
| `get_agent(ctx, name)` | `get_agent(context, agent_name)` | Get agent proxy in orchestration |

### Example 1 — minimal serverless agent app

```python
# function_app.py
from agent_framework.azure import AgentFunctionApp
from agent_framework.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient()

support_agent = client.as_agent(
    name="SupportAgent",
    instructions="You are a helpful customer support agent.",
)

# Creates HTTP trigger at /api/SupportAgent and durable entity
app = AgentFunctionApp(agents=[support_agent])
```

### Example 2 — multiple agents with selective HTTP endpoints

```python
from agent_framework.azure import AgentFunctionApp
from agent_framework.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient()

public_agent = client.as_agent(
    name="PublicAgent",
    instructions="Answer general questions.",
)
internal_agent = client.as_agent(
    name="InternalAgent",
    instructions="Handle sensitive internal operations.",
)

app = AgentFunctionApp(http_auth_level="admin")
# Public agent gets HTTP endpoint; internal agent exposed only via orchestration
app.add_agent(public_agent, http_endpoint_enabled=True, mcp_tool_enabled=False)
app.add_agent(internal_agent, http_endpoint_enabled=False, mcp_tool_enabled=False)
```

### Example 3 — orchestration with agent fan-out

```python
from agent_framework.azure import AgentFunctionApp
from agent_framework.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient()

analyst = client.as_agent(name="Analyst", instructions="Analyse financial data.")
writer  = client.as_agent(name="Writer",  instructions="Write executive summaries.")

app = AgentFunctionApp(agents=[analyst, writer])


@app.orchestration_trigger(context_name="context")
def report_orchestration(context):
    analyst_proxy = app.get_agent(context, "Analyst")
    writer_proxy  = app.get_agent(context, "Writer")

    analyst_session = analyst_proxy.create_session()
    analysis_task   = analyst_proxy.run(
        "Analyse Q3 revenue data and identify trends.", session=analyst_session
    )
    analysis = yield analysis_task

    writer_session  = writer_proxy.create_session()
    summary_task    = writer_proxy.run(
        f"Write an executive summary for:\n{analysis.text}", session=writer_session
    )
    summary = yield summary_task
    return summary.text
```

### Example 4 — workflow hosting with HITL

```python
from agent_framework import WorkflowBuilder
from agent_framework.azure import AgentFunctionApp
from agent_framework.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient()
review_agent   = client.as_agent(name="Reviewer",  instructions="Review and approve requests.")
approval_agent = client.as_agent(name="Approver",  instructions="Final approval authority.")

# Build a workflow with human-in-the-loop
# workflow = WorkflowBuilder(...).build()

# Pass the workflow — the app auto-registers the constituent agents
# app = AgentFunctionApp(workflow=workflow)
```

### Example 5 — MCP tool trigger for tool-as-a-service

```python
from agent_framework.azure import AgentFunctionApp
from agent_framework.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient()
code_agent = client.as_agent(
    name="CodeAgent",
    instructions="Write and review Python code.",
)

# enable_mcp_tool_trigger=True creates an MCP tool endpoint for the agent
app = AgentFunctionApp()
app.add_agent(code_agent, http_endpoint_enabled=True, mcp_tool_enabled=True)
# The agent is now callable as an MCP tool from Copilot Studio or VS Code
```

---

## Volume index

| Vol. | Version | Classes |
|------|---------|---------|
| [1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) | 1.6.0 | `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool` |
| [2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) | 1.6.0 | `FileHistoryProvider`, `AgentMiddleware`, `ChatMiddleware`, `FunctionMiddleware`, `CompactionProvider`, `ToolResultCompactionStrategy`, `TokenBudgetComposedStrategy`, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult` |
| [3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) | 1.7.0 | `BackgroundAgentsProvider`, `MemoryContextProvider`, `TodoProvider`, `AgentModeProvider`, `SummarizationStrategy`, `ContextWindowCompactionStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`, `WorkflowViz`, `MCPStreamableHTTPTool` + `MCPWebsocketTool` |
| [4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) | 1.7.0 | `Message` + `Content`, `ChatOptions` + `ChatResponse`, `ResponseStream`, `AgentContext`, `FunctionalWorkflow` + `StepWrapper`, `WorkflowEvent`, `SkillsSource`, `EvalItem` + `EvalResults`, `TokenizerProtocol`, `ConversationSplit` |
| [5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) | 1.7.0 | `Executor` + `@handler` + `@executor`, `AgentExecutor`, edge groups, `Runner`, `SessionContext`, `AgentSession`, `BaseChatClient`, `SecretString`, `WorkflowCheckpoint`, exceptions |
| [6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) | 1.7.0 | `ExperimentalFeature`, `WorkflowRunState`, `WorkflowExecutor`, `AgentResponse`, `BaseEmbeddingClient`, `FunctionInvocationConfiguration`, `ClassSkill`, `Annotation`, capability protocols, middleware layers |
| [7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) | 1.7.0 | `ContextProvider`, `BackgroundTaskInfo`, `GroupChatBuilder`, `HandoffBuilder`, `MagenticBuilder`, `SequentialBuilder`, `ConcurrentBuilder`, `AgentFactory`, `WorkflowFactory`, `SecureAgentConfig`, `FunctionalWorkflowAgent`, `ObservabilitySettings` |
| [8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) | 1.8.0 | `AgentFileStore`, `FileAccessProvider`, `MCPSkill`, `ToolMode`, `AgentEvalConverter`, `ChatContext`, `WorkflowAgent`, `TruncationStrategy`, `HistoryProvider`, `DelegatingSkillsSource` |
| [9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) | 1.8.0 | `OllamaChatClient`, `PurviewPolicyMiddleware`, `DurableAIAgent` + `DurableAIAgentClient` + `DurableAIAgentWorker`, `GitHubCopilotAgent`, `HyperlightExecuteCodeTool` + `HyperlightCodeActProvider`, `Mem0ContextProvider`, `RedisContextProvider` + `RedisHistoryProvider`, `StandardMagenticManager`, `FileSkillsSource` |
| [10](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10/) | 1.8.0 | `A2AAgent` + `A2AAgentSession`, `A2AExecutor`, `AGUIChatClient`, `AnthropicClient` + `ThinkingConfig`, `BedrockChatClient`, `CopilotStudioAgent`, `GroupChatOrchestrator`, `AgentBasedGroupChatOrchestrator`, `HandoffBuilder`, `ExternalInputRequest` |
| **11** | **1.8.1** | **`OpenAIChatClient`, `OpenAIChatCompletionClient`, `FoundryChatClient`, `FoundryMemoryProvider`, `FoundryLocalClient`, `AzureAISearchContextProvider`, `CosmosCheckpointStorage` + `CosmosHistoryProvider`, `OrchestrationState`, `ClaudeAgent`, `AgentFunctionApp`** |

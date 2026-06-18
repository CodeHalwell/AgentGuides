---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 16"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.8.1: FoundryAgent+FoundryAgentOptions (hosted Azure AI agents), FoundryLocalClient (on-device LLM), FoundryMemoryProvider (Foundry semantic memory), FoundryEvals+GeneratedEvaluatorRef (cloud eval harness — 19 built-in evaluators), BedrockChatClient+BedrockChatOptions+BedrockGuardrailConfig (Bedrock Converse API), BedrockEmbeddingClient+BedrockEmbeddingOptions (Bedrock Titan embeddings), MagenticManagerBase (custom Magentic manager authoring), BaseGroupChatOrchestrator+GroupChatRequestSentEvent+GroupChatResponseReceivedEvent (group chat base class and event model), AgentRequestInfoResponse+CacheProvider (HITL approval response + cache protocol), Purview exception hierarchy + acquire_token (MSAL token flow)."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 39
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 16

Verified against **agent-framework 1.8.1** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source. Sub-packages introspected: `agent_framework.foundry`, `agent_framework.amazon`,
`agent_framework.orchestrations`, `agent_framework.microsoft`.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, middleware ABCs, compaction, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — harness providers, compaction strategies, `WorkflowViz`, MCP transports
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — message/chat types, `ResponseStream`, `AgentContext`, functional workflows, `SkillsSource`, eval model, tokenizer, `ConversationSplit`
- [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) — `Executor`, `AgentExecutor`, edge groups, `Runner`, `SessionContext`, `AgentSession`, `BaseChatClient`, `SecretString`, `WorkflowCheckpoint`, exceptions
- [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) — feature staging, `WorkflowRunState`, `WorkflowExecutor`, `AgentResponse`, embedding clients, `FunctionInvocationConfiguration`, `ClassSkill`, `Annotation`, capability protocols, middleware layers
- [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) — `ContextProvider`, `BackgroundTaskInfo`, orchestration builders, `AgentFactory`, `SecureAgentConfig`, `ObservabilitySettings`
- [Vol. 8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) — file store hierarchy, `FileAccessProvider`, `MCPSkill`, `ToolMode`, eval helpers, `ChatContext`, `WorkflowAgent`, compaction, history providers, skills composition
- [Vol. 9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) — `OllamaChatClient`, `PurviewPolicyMiddleware`, `DurableAIAgent`, `GitHubCopilotAgent`, `HyperlightExecuteCodeTool`, `Mem0ContextProvider`, Redis providers, Magentic internals, `FileSkillsSource`
- [Vol. 10](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10/) — `Workflow`, `InProcRunnerContext`, `FunctionExecutor`, `FunctionInvocationLayer`, memory harness, todo harness, `DeduplicatingSkillsSource`, `SkillsProvider`, `MCPTaskOptions`, `InMemoryCheckpointStorage`, `BaseAgent`
- [Vol. 11](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v11/) — telemetry layers, `Edge`+`EdgeGroup` primitives, `Case`+`Default`, `EdgeRunner` hierarchy, `ExecutionContext`, `WorkflowGraphValidator`, `MCPTool`, serialization mixin, `Evaluator`, `PerServiceCallHistoryPersistingMiddleware`
- [Vol. 12](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v12/) — Skills ABCs, `FileSkill`, `InlineSkillResource`+`InlineSkillScript`, `FileSkillScript`+`SkillScriptRunner`, `SupportsAgentRun`, `RunnerContext`, edge-routing descriptors, `WorkflowValidationError` hierarchy, `A2AAgent`+`A2AExecutor`, exception leaf classes
- [Vol. 13](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v13/) — OpenAI Responses/Completions/Embedding clients, Anthropic + Claude agent clients, multi-cloud Claude variants, group-chat + handoff + Magentic orchestration internals, declarative HTTP/MCP/approval handlers
- [Vol. 14](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v14/) — `State` (superstep cache), `OutputDesignation`, `MessageType`+`WorkflowMessage` internals, `DictConvertible` mixin, middleware pipeline hierarchy, `MiddlewareDict`, `FunctionRequestResult`, `OtelAttr`, security policy classes
- [Vol. 15](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v15/) — AG-UI client layer, AG-UI protocol wrappers, ChatKit, DevServer, GAIA benchmark, CopilotStudioAgent, AzureAISearchContextProvider, CosmosHistoryProvider, Durable external layer, AgentFunctionApp

This volume covers **ten new class groups** focused on the Azure Foundry hosting surface,
Amazon Bedrock providers, orchestration base classes, and enterprise integration utilities
that have never been documented in any previous volume.

---

## Table of contents

1. [`FoundryAgent` + `FoundryAgentOptions` + `RawFoundryAgent` + `RawFoundryAgentChatClient`](#1-foundryagent-and-foundryagentoptions)
2. [`FoundryLocalClient` + `FoundryLocalChatOptions` + `FoundryLocalSettings`](#2-foundrylocalclient)
3. [`FoundryMemoryProvider`](#3-foundry-memory-provider)
4. [`FoundryEvals` + `GeneratedEvaluatorRef` + `evaluate_foundry_target` + `evaluate_traces`](#4-foundryevals)
5. [`BedrockChatClient` + `BedrockChatOptions` + `BedrockGuardrailConfig` + `BedrockSettings`](#5-bedrockchatclient)
6. [`BedrockEmbeddingClient` + `BedrockEmbeddingOptions` + `BedrockEmbeddingSettings`](#6-bedrockembeddingclient)
7. [`MagenticManagerBase`](#7-magenticmanagerbase)
8. [`BaseGroupChatOrchestrator` + `GroupChatRequestSentEvent` + `GroupChatResponseReceivedEvent`](#8-basegroupchatorchestrator-and-events)
9. [`AgentRequestInfoResponse` + `CacheProvider`](#9-agentrequestinforesponse-cacheprovider)
10. [Purview exception hierarchy + `acquire_token`](#10-purview-exception-hierarchy-acquire_token)

---

## 1. `FoundryAgent` and `FoundryAgentOptions`

**Module:** `agent_framework.foundry`  
**Install:** `pip install agent-framework[foundry]`

`FoundryAgent` connects to an **existing** PromptAgent or HostedAgent deployed in
Azure AI Foundry — the framework authenticates with the Foundry service, receives a
streaming run response, and locally executes any function tools. It is the preferred
approach when your agents are already configured, versioned, and deployed in the Foundry
portal.

### Class hierarchy

```
RawFoundryAgent[FoundryAgentOptionsT]
  └── FoundryAgent (adds AgentMiddlewareLayer + AgentTelemetryLayer)
```

`RawFoundryAgentChatClient` is the underlying chat client that the agent creates
internally — you can pass `client_type=RawFoundryAgentChatClient` to bypass
middleware and telemetry on the model call.

### Constructor reference

```
FoundryAgent(
    *,
    project_endpoint: str | None = None,    # or FOUNDRY_PROJECT_ENDPOINT env var
    agent_name: str | None = None,          # name of the Foundry agent
    agent_version: str | None = None,       # required for PromptAgents; omit for HostedAgents
    credential: AzureCredentialTypes | None = None,
    project_client: AIProjectClient | None = None,  # bring-your-own client
    allow_preview: bool | None = None,      # enable preview session APIs for HostedAgents
    default_headers: Mapping[str, str] | None = None,
    tools: ... | None = None,              # FunctionTool / callable / sequence
    context_providers: Sequence[ContextProvider] | None = None,
    middleware: Sequence[MiddlewareTypes] | None = None,
    client_type: type[RawFoundryAgentChatClient] | None = None,
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
    id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    instructions: str | None = None,
    default_options: FoundryAgentOptionsT | Mapping[str, Any] | None = None,
    require_per_service_call_history_persistence: bool = False,
    function_invocation_configuration: FunctionInvocationConfiguration | None = None,
    compaction_strategy: CompactionStrategy | None = None,
    tokenizer: TokenizerProtocol | None = None,
    additional_properties: Mapping[str, Any] | None = None,
    timeout: float | None = None,
)
```

### `FoundryAgentOptions`

Extends `OpenAIChatOptions` with two Foundry-specific fields:

| Field | Type | Purpose |
|-------|------|---------|
| `extra_body` | `dict[str, Any]` | Additional JSON merged into the Responses API request body |
| `isolation_key` | `str` | Isolation key for lazily creating a HostedAgent session via `project_client.beta.agents.create_session(...)` |

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| PromptAgents | Always require `agent_version`; the framework pins the exact version |
| HostedAgents | Omit `agent_version`; set `allow_preview=True` to use preview session APIs |
| `project_client` | If you pass your own `AIProjectClient`, `allow_preview` is ignored |
| `client_type` override | Pass `client_type=RawFoundryAgentChatClient` to get a raw client without middleware layers on the chat call |
| Function tools | In **preview mode** (`allow_preview=True`) tool definitions are forwarded with the request; in **non-preview mode** they are **stripped** from the request body — the service only sees tools declared in the deployed Foundry agent definition. Local `FunctionTool` objects are still executed client-side by the function invocation layer when the service returns a tool-call response |
| Auth | Accepts any `AzureCredentialTypes` — `DefaultAzureCredential`, `AzureCliCredential`, managed identity |

**Example 1 — call a PromptAgent with a pinned version:**

```python
import asyncio
from agent_framework.foundry import FoundryAgent
from azure.identity import DefaultAzureCredential

async def main():
    agent = FoundryAgent(
        project_endpoint="https://my-proj.services.ai.azure.com",
        agent_name="customer-support-v2",
        agent_version="1.0",
        credential=DefaultAzureCredential(),
    )
    result = await agent.run("How do I reset my password?")
    print(result.text)

asyncio.run(main())
```

**Example 2 — call a HostedAgent with preview session APIs and an isolation key:**

```python
import asyncio
from agent_framework import AgentSession
from agent_framework.foundry import FoundryAgent, FoundryAgentOptions
from azure.identity import AzureCliCredential

async def main():
    agent = FoundryAgent(
        project_endpoint="https://my-proj.services.ai.azure.com",
        agent_name="my-hosted-agent",
        credential=AzureCliCredential(),
        allow_preview=True,
        default_options=FoundryAgentOptions(
            isolation_key="tenant-abc",   # creates one hosted session per tenant
        ),
    )
    # AgentSession is required: isolation_key only calls create_session() when
    # a session object is present (session.service_session_id is set on first run
    # and reused on subsequent runs for the same session).
    session = AgentSession()
    result = await agent.run("Summarise my account activity", session=session)
    print(result.text)

asyncio.run(main())
```

**Example 3 — FoundryAgent with local function tools and middleware:**

```python
import asyncio
from agent_framework import tool
from agent_framework.foundry import FoundryAgent
from azure.identity import DefaultAzureCredential

@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"Sunny, 22°C in {city}"

async def main():
    agent = FoundryAgent(
        project_endpoint="https://my-proj.services.ai.azure.com",
        agent_name="weather-assistant",
        agent_version="2.1",
        credential=DefaultAzureCredential(),
        tools=[get_weather],
    )
    result = await agent.run("What's the weather in Tokyo?")
    print(result.text)

asyncio.run(main())
```

---

## 2. `FoundryLocalClient`

**Module:** `agent_framework.foundry`  
**Install:** `pip install agent-framework[foundry-local]`

`FoundryLocalClient` brings **on-device LLM inference** via the
[Foundry Local SDK](https://github.com/microsoft/foundry-local) — no cloud round-trip,
no API key. It is a full chat client with the standard middleware + telemetry stack.

### Class hierarchy

```
RawOpenAIChatCompletionClient[FoundryLocalChatOptionsT]  (OpenAI Completions compatible)
  ├── ChatTelemetryLayer
  ├── ChatMiddlewareLayer
  ├── FunctionInvocationLayer
  └── FoundryLocalClient
```

### Constructor reference

```
FoundryLocalClient(
    model: str | None = None,          # model alias or HF repo; resolved via FOUNDRY_LOCAL_MODEL
    *,
    bootstrap: bool = True,            # start the Foundry Local service if not running
    timeout: float | None = None,
    prepare_model: bool = True,        # download + prepare model before first call
    device: DeviceType | None = None,  # "cpu" | "gpu" | "npu" | None (auto-detect)
    additional_properties: dict[str, Any] | None = None,
    middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
    function_invocation_configuration: FunctionInvocationConfiguration | None = None,
    env_file_path: str | None = None,
    env_file_encoding: str = "utf-8",
)
```

### `FoundryLocalSettings`

Loaded from environment with prefix `FOUNDRY_LOCAL_`:

| Setting | Env var | Description |
|---------|---------|-------------|
| `model` | `FOUNDRY_LOCAL_MODEL` | Default model alias |

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| `bootstrap=True` | Automatically starts the `foundry-local` background service on the first client instantiation |
| `prepare_model=True` | Downloads and quantises the model on first use; subsequent calls skip this |
| `device` | `None` auto-selects the fastest available accelerator; `"npu"` for Copilot+ PCs |
| OpenAI wire format | Uses the Chat Completions endpoint — `stream=True` and `response_format` (Pydantic structured output) both work |
| Env fallback | Model resolved from `FOUNDRY_LOCAL_MODEL` when not passed at construction time |

**Example 1 — minimal local agent (no cloud dependency):**

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryLocalClient

async def main():
    client = FoundryLocalClient(model="phi-4-mini")
    agent = Agent(
        client=client,
        instructions="You are a helpful offline assistant.",
    )
    result = await agent.run("Explain quantum entanglement in one paragraph.")
    print(result.text)

asyncio.run(main())
```

**Example 2 — use GPU acceleration with custom function tools:**

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.foundry import FoundryLocalClient
from foundry_local.models import DeviceType

@tool
def word_count(text: str) -> int:
    """Return the number of words in text."""
    return len(text.split())

async def main():
    client = FoundryLocalClient(
        model="phi-4-mini",
        device=DeviceType.GPU,
        prepare_model=True,
    )
    agent = Agent(client=client, tools=[word_count])
    result = await agent.run("How many words are in 'the quick brown fox'?")
    print(result.text)

asyncio.run(main())
```

**Example 3 — structured output from a local model:**

```python
import asyncio
from pydantic import BaseModel
from agent_framework import Agent, ChatOptions
from agent_framework.foundry import FoundryLocalClient

class Sentiment(BaseModel):
    label: str
    confidence: float
    reasoning: str

async def main():
    client = FoundryLocalClient(model="phi-4-mini")
    agent = Agent(client=client)
    result = await agent.run(
        "Classify: 'The service was excellent!'",
        options=ChatOptions(response_format=Sentiment),
    )
    sentiment: Sentiment = result.value
    print(f"{sentiment.label} ({sentiment.confidence:.0%}): {sentiment.reasoning}")

asyncio.run(main())
```

---

## 3. Foundry Memory Provider

**Module:** `agent_framework.foundry`  
**Install:** `pip install agent-framework[foundry]`

`FoundryMemoryProvider` integrates Azure AI Foundry's managed **Memory Store** as a
`ContextProvider`. On each `before_run`, it searches for memories relevant to the current
turn and prepends them to the agent's context. On `after_run`, it asynchronously stores
the turn as a new memory, applying a configurable delay to batch rapid interactions.

### Constructor reference

```
FoundryMemoryProvider(
    source_id: str = "foundry_memory",
    *,
    project_client: AIProjectClient | None = None,
    project_endpoint: str | None = None,       # or FOUNDRY_PROJECT_ENDPOINT env var
    credential: AzureCredentialTypes | None = None,
    allow_preview: bool | None = None,
    memory_store_name: str,                    # required — Foundry Memory Store name
    scope: str | None = None,                  # namespace, e.g. user ID; falls back to session_id
    context_prompt: str | None = None,         # header injected above retrieved memories
    update_delay: int = 300,                   # seconds before the async memory write fires
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
)
```

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| `DEFAULT_SOURCE_ID` | `"foundry_memory"` |
| `DEFAULT_CONTEXT_PROMPT` | `"## Memories\nConsider the following memories when answering user questions:"` |
| `scope=None` | Falls back to `session.session_id` — one memory namespace per conversation |
| `update_delay=300` | Waits 5 minutes before writing; set to `0` for immediate persistence |
| Lifecycle | Implements `before_run` (search + inject) and `after_run` (async write) |
| Authentication | Accepts `project_client` directly, or builds one from `project_endpoint` + `credential` |
| Async context manager | `async with FoundryMemoryProvider(...) as mp` closes the underlying `AIProjectClient` |

**Example 1 — user-scoped persistent memory:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryMemoryProvider, FoundryChatClient
from azure.identity import DefaultAzureCredential

async def main():
    memory = FoundryMemoryProvider(
        project_endpoint="https://my-proj.services.ai.azure.com",
        credential=DefaultAzureCredential(),
        memory_store_name="user-memories",
        scope="user-42",
    )
    async with memory:
        agent = Agent(
            client=FoundryChatClient(model="gpt-4o"),
            context_providers=[memory],
            instructions="You are a personal assistant with memory.",
        )
        # Turn 1: agent stores a memory about the user's preference
        await agent.run("I prefer concise answers.")
        # Turn 2: memory is retrieved and injected before calling the LLM
        result = await agent.run("Summarise the main benefits of Python.")
        print(result.text)

asyncio.run(main())
```

**Example 2 — immediate write with custom scope and prompt:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryMemoryProvider, FoundryChatClient
from azure.identity import AzureCliCredential

async def main():
    memory = FoundryMemoryProvider(
        project_endpoint="https://my-proj.services.ai.azure.com",
        credential=AzureCliCredential(),
        memory_store_name="session-memories",
        scope="session-abc123",
        context_prompt="## Recalled context\nUse the following for context:",
        update_delay=0,  # write immediately after each turn
    )
    async with memory:
        agent = Agent(
            client=FoundryChatClient(model="gpt-4o"),
            context_providers=[memory],
        )
        result = await agent.run("What colour is the sky?")
        print(result.text)

asyncio.run(main())
```

**Example 3 — bring-your-own AIProjectClient:**

```python
import asyncio
from azure.ai.projects.aio import AIProjectClient
from azure.identity import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.foundry import FoundryMemoryProvider, FoundryChatClient

async def main():
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(
        endpoint="https://my-proj.services.ai.azure.com",
        credential=credential,
    )
    async with project_client:
        memory = FoundryMemoryProvider(
            project_client=project_client,
            memory_store_name="shared-memories",
            scope="team-engineering",
        )
        agent = Agent(
            client=FoundryChatClient(model="gpt-4o"),
            context_providers=[memory],
        )
        result = await agent.run("What are our team's current priorities?")
        print(result.text)

asyncio.run(main())
```

---

## 4. `FoundryEvals`

**Module:** `agent_framework.foundry`  
**Install:** `pip install agent-framework[foundry]`  
**Feature stage:** `@experimental(feature_id=ExperimentalFeature.EVALS)`

`FoundryEvals` implements the `Evaluator` protocol backed by Azure AI Foundry's built-in
evaluation service. Pass it to `evaluate_agent()` or `evaluate_workflow()` just like any
other evaluator — internally it creates an eval definition via the OpenAI Evals API and
polls until completion.

### Constructor reference

```
FoundryEvals(
    *,
    client: FoundryChatClient | None = None,        # auto-created from env vars if omitted
    project_client: AIProjectClient | None = None,   # alternative to client
    model: str | None = None,                        # evaluator LLM; resolved from client.model
    evaluators: Sequence[str | GeneratedEvaluatorRef] | None = None,  # None = smart defaults
    conversation_split: ConversationSplitter = ConversationSplit.LAST_TURN,
    poll_interval: float = 5.0,                      # seconds between status polls
    timeout: float = 180.0,                          # max wait seconds
)
```

### Built-in evaluator name constants (source-verified at 1.8.1)

All are `str` class attributes — use them to avoid typos:

| Constant | Value | Category |
|----------|-------|----------|
| `INTENT_RESOLUTION` | `"intent_resolution"` | Agent behavior |
| `TASK_ADHERENCE` | `"task_adherence"` | Agent behavior |
| `TASK_COMPLETION` | `"task_completion"` | Agent behavior |
| `TASK_NAVIGATION_EFFICIENCY` | `"task_navigation_efficiency"` | Agent behavior |
| `TOOL_CALL_ACCURACY` | `"tool_call_accuracy"` | Tool usage |
| `TOOL_SELECTION` | `"tool_selection"` | Tool usage |
| `TOOL_INPUT_ACCURACY` | `"tool_input_accuracy"` | Tool usage |
| `TOOL_OUTPUT_UTILIZATION` | `"tool_output_utilization"` | Tool usage |
| `TOOL_CALL_SUCCESS` | `"tool_call_success"` | Tool usage |
| `COHERENCE` | `"coherence"` | Quality |
| `FLUENCY` | `"fluency"` | Quality |
| `RELEVANCE` | `"relevance"` | Quality |
| `GROUNDEDNESS` | `"groundedness"` | Quality |
| `RESPONSE_COMPLETENESS` | `"response_completeness"` | Quality |
| `SIMILARITY` | `"similarity"` | Quality |
| `VIOLENCE` | `"violence"` | Safety |
| `SEXUAL` | `"sexual"` | Safety |
| `SELF_HARM` | `"self_harm"` | Safety |
| `HATE_UNFAIRNESS` | `"hate_unfairness"` | Safety |

### `GeneratedEvaluatorRef`

A reference to a rubric evaluator already stored in a Foundry project. Pass instances to
`FoundryEvals(evaluators=[...])` to score with a pre-existing rubric; the SDK does not
create or modify the evaluator definition, only references it by name.

```
GeneratedEvaluatorRef(
    name: str,                  # evaluator name as stored in Foundry, e.g. "reservation-policy-rubric"
    version: str | None = None, # pinned version; None resolves to latest (emits a warning)
    display_name: str | None = None,  # human-readable label in result summaries
)
# Convenience factory for explicit latest-resolution (discouraged in CI):
GeneratedEvaluatorRef.latest(name: str, *, display_name: str | None = None)
```

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| Smart defaults | When `evaluators=None`, uses `relevance`, `coherence`, and `task_adherence`; automatically adds `tool_call_accuracy` when items contain tool definitions |
| `builtin.*` prefix | Foundry built-ins use the `builtin.` prefix internally; the constants above resolve to the correct names |
| `poll_interval` | Default `5.0` — reduce for faster CI, increase to avoid rate limits on large eval batches |
| Zero-config | Reads `FOUNDRY_PROJECT_ENDPOINT` and `FOUNDRY_MODEL` env vars to auto-create the `FoundryChatClient` |

**Example 1 — evaluate an agent with default evaluators:**

```python
import asyncio
from agent_framework import Agent, evaluate_agent
from agent_framework.foundry import FoundryEvals, FoundryChatClient

async def main():
    client = FoundryChatClient(model="gpt-4o")
    agent = Agent(client=client, instructions="You are a helpful assistant.")

    # evaluate_agent accepts plain strings for queries and expected_output
    evals = FoundryEvals(client=client)  # uses relevance + coherence + task_adherence
    results = await evaluate_agent(
        agent=agent,
        queries=["What is 2+2?", "Capital of France?"],
        expected_output=["4", "Paris"],
        evaluators=evals,
    )
    # evaluate_agent returns list[EvalResults]; one entry per evaluator provider
    for r in results:
        print(f"{r.provider}: passed={r.passed} failed={r.failed} total={r.total}")

asyncio.run(main())
```

**Example 2 — select specific evaluators including safety checks:**

```python
import asyncio
from agent_framework import Agent, evaluate_agent
from agent_framework.foundry import FoundryEvals, FoundryChatClient

async def main():
    client = FoundryChatClient(model="gpt-4o")
    agent = Agent(client=client)

    evals = FoundryEvals(
        client=client,
        evaluators=[
            FoundryEvals.FLUENCY,       # no context field required
            FoundryEvals.RELEVANCE,
            FoundryEvals.HATE_UNFAIRNESS,
            FoundryEvals.VIOLENCE,
        ],
        poll_interval=3.0,
        timeout=120.0,
    )
    results = await evaluate_agent(
        agent=agent,
        queries=["Tell me about climate change."],
        evaluators=evals,
    )
    for r in results:
        print(f"{r.provider}: {r.passed}/{r.total}")
        if r.per_evaluator:
            for eval_name, counts in r.per_evaluator.items():
                print(f"  {eval_name}: {counts}")

asyncio.run(main())
```

**Example 3 — tool-call accuracy eval for a function-calling agent:**

```python
import asyncio
from agent_framework import Agent, tool, evaluate_agent
from agent_framework.foundry import FoundryEvals, FoundryChatClient

@tool
def get_stock_price(ticker: str) -> float:
    """Return the current stock price."""
    return 42.0

async def main():
    client = FoundryChatClient(model="gpt-4o")
    agent = Agent(client=client, tools=[get_stock_price])

    evals = FoundryEvals(
        client=client,
        evaluators=[
            FoundryEvals.TOOL_CALL_ACCURACY,
            FoundryEvals.TOOL_SELECTION,
            FoundryEvals.TOOL_INPUT_ACCURACY,
        ],
    )
    results = await evaluate_agent(
        agent=agent,
        queries=["What is the stock price of MSFT?"],
        expected_output=["42.0"],
        evaluators=evals,
    )
    for r in results:
        print(f"Tool call eval — {r.provider}: {r.passed}/{r.total}")

asyncio.run(main())
```

---

## 5. `BedrockChatClient`

**Module:** `agent_framework.amazon`  
**Install:** `pip install agent-framework[bedrock]`

`BedrockChatClient` wraps Amazon Bedrock's [Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
with the standard framework middleware + telemetry stack. It works with any
Bedrock-supported foundation model (Claude, Titan, Llama, Command, etc.) through a
single unified API.

### Constructor reference

```
BedrockChatClient(
    *,
    region: str | None = None,             # or BEDROCK_REGION env var; defaults to "us-east-1"
    model: str | None = None,              # or BEDROCK_CHAT_MODEL env var
    access_key: str | None = None,         # or BEDROCK_ACCESS_KEY env var
    secret_key: str | None = None,         # or BEDROCK_SECRET_KEY env var
    session_token: str | None = None,      # or BEDROCK_SESSION_TOKEN env var
    client: BaseClient | None = None,      # pre-built boto3 bedrock-runtime client
    boto3_session: Boto3Session | None = None,
    additional_properties: dict[str, Any] | None = None,
    middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
    function_invocation_configuration: FunctionInvocationConfiguration | None = None,
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
)
```

### `BedrockChatOptions`

Extends `ChatOptions` with Bedrock-specific fields. Options that are **not supported** in
the Converse API are declared as `None` typed, so passing them raises a type error at
the call site:

| Bedrock-specific field | Type | Purpose |
|------------------------|------|---------|
| `guardrailConfig` | `BedrockGuardrailConfig` | Content filtering and safety guardrails |
| `performanceConfig` | `dict[str, Any]` | Latency optimization settings |
| `requestMetadata` | `dict[str, str]` | Key-value metadata (max 2048 chars total) |
| `promptVariables` | `dict[str, dict[str, str]]` | Variables for managed prompts |

**Unsupported options** (type-annotated `None`): `seed`, `frequency_penalty`,
`presence_penalty`, `allow_multiple_tool_calls`, `user`, `store`, `logit_bias`, `metadata`.

### `BedrockGuardrailConfig`

> **Note (1.8.1):** `BedrockGuardrailConfig` and the other Bedrock-specific option keys
> (`performanceConfig`, `requestMetadata`, `promptVariables`) are defined in `BedrockChatOptions`
> but are **not forwarded** to the Converse API by `BedrockChatClient._prepare_options` in this
> release. Pass them as native `additional_request_fields` via the underlying boto3 client
> until the SDK surfaces them directly.

```python
class BedrockGuardrailConfig(TypedDict, total=False):
    guardrailIdentifier: str          # guardrail ID
    guardrailVersion: str             # version string, e.g. "DRAFT"
    trace: Literal["enabled", "disabled"]
    streamProcessingMode: Literal["sync", "async"]  # sync blocks stream; async does not
```

### `BedrockSettings` (env prefix `BEDROCK_`)

| Setting key | Env var | Default |
|-------------|---------|---------|
| `region` | `BEDROCK_REGION` | `"us-east-1"` |
| `chat_model` | `BEDROCK_CHAT_MODEL` | — |
| `access_key` | `BEDROCK_ACCESS_KEY` | — |
| `secret_key` | `BEDROCK_SECRET_KEY` | — |
| `session_token` | `BEDROCK_SESSION_TOKEN` | — |

**Example 1 — basic agent on Claude via Bedrock (default IAM credentials):**

```python
import asyncio
from agent_framework import Agent
from agent_framework.amazon import BedrockChatClient

async def main():
    client = BedrockChatClient(
        region="us-east-1",
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    agent = Agent(client=client, instructions="You are a helpful assistant.")
    result = await agent.run("Explain the difference between Bedrock and SageMaker.")
    print(result.text)

asyncio.run(main())
```

**Example 2 — multi-turn conversation with explicit per-call options:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.amazon import BedrockChatClient

async def main():
    client = BedrockChatClient(
        region="us-east-1",
        model="amazon.nova-pro-v1:0",
    )
    agent = Agent(
        client=client,
        instructions="You are a concise technical writer.",
        default_options={"max_tokens": 512, "temperature": 0.3},
    )
    # First turn — establish context
    r1 = await agent.run("What is the Bedrock Converse API?")
    print("Turn 1:", r1.text)

    # Second turn — follow-up using the same session (history is maintained)
    r2 = await agent.run("How does it compare to InvokeModel?")
    print("Turn 2:", r2.text)

asyncio.run(main())
```

**Example 3 — explicit AWS credentials for cross-account assume-role:**

```python
import asyncio
import boto3
from agent_framework import Agent
from agent_framework.amazon import BedrockChatClient

async def main():
    # Assume a cross-account role and extract temporary credentials
    sts = boto3.client("sts")
    role = sts.assume_role(
        RoleArn="arn:aws:iam::123456789012:role/BedrockAccessRole",
        RoleSessionName="agent-session",
    )
    creds = role["Credentials"]

    client = BedrockChatClient(
        region="us-west-2",
        model="meta.llama3-70b-instruct-v1:0",
        access_key=creds["AccessKeyId"],
        secret_key=creds["SecretAccessKey"],
        session_token=creds["SessionToken"],
    )
    agent = Agent(client=client)
    result = await agent.run("Summarise our Q3 financial performance.")
    print(result.text)

asyncio.run(main())
```

---

## 6. `BedrockEmbeddingClient`

**Module:** `agent_framework.amazon`  
**Install:** `pip install agent-framework[bedrock]`  
**OTel provider name:** `"aws.bedrock"` (same as `BedrockChatClient`)

`BedrockEmbeddingClient` wraps Amazon Bedrock's `invoke_model` API for embedding
generation, defaulting to the **Amazon Titan Embeddings v2** family. It layers
`EmbeddingTelemetryLayer` on top for automatic OTel span emission.

### Constructor reference

```
BedrockEmbeddingClient(
    *,
    region: str | None = None,              # or BEDROCK_REGION; defaults "us-east-1"
    model: str | None = None,               # or BEDROCK_EMBEDDING_MODEL
    access_key: str | None = None,
    secret_key: str | None = None,
    session_token: str | None = None,
    client: BaseClient | None = None,
    boto3_session: Boto3Session | None = None,
    otel_provider_name: str | None = None,
    additional_properties: dict[str, Any] | None = None,
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
)
```

### `BedrockEmbeddingOptions`

Extends `EmbeddingGenerationOptions` with one Bedrock-specific field:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `normalize` | `bool` | — | When `True`, Titan normalises embedding vectors to unit length |

### `BedrockEmbeddingSettings` (env prefix `BEDROCK_`)

| Setting key | Env var |
|-------------|---------|
| `region` | `BEDROCK_REGION` |
| `embedding_model` | `BEDROCK_EMBEDDING_MODEL` |
| `access_key` | `BEDROCK_ACCESS_KEY` |
| `secret_key` | `BEDROCK_SECRET_KEY` |
| `session_token` | `BEDROCK_SESSION_TOKEN` |

**Example 1 — generate embeddings with Titan v2:**

```python
import asyncio
from agent_framework.amazon import BedrockEmbeddingClient

async def main():
    client = BedrockEmbeddingClient(
        model="amazon.titan-embed-text-v2:0",
        region="us-east-1",
    )
    result = await client.get_embeddings(["Machine learning", "Deep learning"])
    for emb in result:
        print(f"dim={len(emb.vector)}, first3={emb.vector[:3]}")

asyncio.run(main())
```

**Example 2 — normalised embeddings for cosine similarity search:**

```python
import asyncio
from agent_framework.amazon import BedrockEmbeddingClient, BedrockEmbeddingOptions

async def main():
    client = BedrockEmbeddingClient(model="amazon.titan-embed-text-v2:0")

    options: BedrockEmbeddingOptions = {
        "normalize": True,       # unit-length vectors for dot-product cosine similarity
        "dimensions": 1024,      # Titan v2 supports 256 / 512 / 1024
    }
    texts = ["what is retrieval augmented generation?", "RAG combines retrieval with LLMs"]
    result = await client.get_embeddings(texts, options=options)
    # cosine similarity = dot product (vectors already unit length)
    import math
    v1, v2 = result[0].vector, result[1].vector
    similarity = sum(a * b for a, b in zip(v1, v2))
    print(f"Cosine similarity: {similarity:.4f}")

asyncio.run(main())
```

**Example 3 — plug into `MemoryContextProvider` for semantic recall:**

```python
import asyncio
from agent_framework import Agent
from agent_framework import MemoryContextProvider, MemoryFileStore
from agent_framework.amazon import BedrockChatClient, BedrockEmbeddingClient

async def main():
    embedding_client = BedrockEmbeddingClient(
        model="amazon.titan-embed-text-v2:0",
        region="us-east-1",
    )
    chat_client = BedrockChatClient(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region="us-east-1",
    )
    # Demonstrate get_embeddings directly
    results = await embedding_client.get_embeddings(
        ["The project deadline is June 30.", "Budget is $50k."]
    )
    # GeneratedEmbeddings subclasses list[Embedding] — index directly, no .embeddings attribute
    print(f"Embedded {len(results)} strings, dim={len(results[0].vector)}")

    # Pair with file-backed persistent memory (MemoryFileStore, not a vector store)
    file_store = MemoryFileStore(
        base_path="/tmp/bedrock-memories",
        owner_state_key="user_id",     # session-state key holding the current user/owner ID
    )
    memory = MemoryContextProvider(store=file_store)
    agent = Agent(
        client=chat_client,
        context_providers=[memory],
        instructions="Use recalled context to answer questions.",
    )
    await agent.run("The project deadline is June 30.")
    result = await agent.run("When is the project due?")
    print(result.text)

asyncio.run(main())
```

---

## 7. `MagenticManagerBase`

**Module:** `agent_framework.orchestrations`  
**Install:** `pip install agent-framework[orchestrations]`

`MagenticManagerBase` is the **abstract base class** for all Magentic-One orchestration
managers. `StandardMagenticManager` (already covered in Vol. 9) is the built-in
implementation; subclass `MagenticManagerBase` to replace the LLM-driven planning with
rule-based, RL-trained, or domain-specific logic.

### Constructor reference

```
MagenticManagerBase(
    *,
    max_stall_count: int = 3,           # consecutive stalls before replanning
    max_reset_count: int | None = None,  # None = unlimited resets
    max_round_count: int | None = None,  # None = unlimited rounds
)
```

### Abstract methods (all must be implemented)

| Method | Signature | Returns |
|--------|-----------|---------|
| `plan` | `async (magentic_context: MagenticContext) -> Message` | The initial task plan as a `Message` |
| `replan` | `async (magentic_context: MagenticContext) -> Message` | A revised plan after stall detection |
| `create_progress_ledger` | `async (magentic_context: MagenticContext) -> MagenticProgressLedger` | A fresh progress ledger for the outer loop |
| `prepare_final_answer` | `async (magentic_context: MagenticContext) -> Message` | The synthesised final response |

### Checkpoint hooks

| Method | Purpose |
|--------|---------|
| `on_checkpoint_save() -> dict[str, Any]` | Serialise custom manager state; default returns `{}` |
| `on_checkpoint_restore(state: dict[str, Any]) -> None` | Restore state from checkpoint; default is no-op |

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| `max_stall_count` | After this many consecutive rounds with no progress, `replan()` is called |
| `max_reset_count` | The maximum number of times the outer loop resets; `None` = unlimited |
| `max_round_count` | Hard ceiling on the number of inner-loop rounds; `None` = unlimited |
| `task_ledger_full_prompt` | Base class exposes this attribute for type safety; concrete managers may override with a `str` class field |
| Wiring | Pass your manager to `MagenticBuilder(manager=my_manager)` |

**Example 1 — minimal rule-based manager for single-step tasks:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.orchestrations import (
    MagenticManagerBase, MagenticBuilder,
    MagenticContext, MagenticProgressLedger, MagenticProgressLedgerItem,
)
from agent_framework import Message
from agent_framework.foundry import FoundryChatClient

class SimpleDelegateManager(MagenticManagerBase):
    """Always delegates to the first available agent — no planning LLM needed."""

    async def plan(self, ctx: MagenticContext) -> Message:
        agents = list(ctx.participant_descriptions.keys())
        return Message(role="assistant", contents=[f"Delegate to: {agents[0]}"])

    async def replan(self, ctx: MagenticContext) -> Message:
        return await self.plan(ctx)

    async def create_progress_ledger(self, ctx: MagenticContext) -> MagenticProgressLedger:
        first_agent = list(ctx.participant_descriptions.keys())[0]
        return MagenticProgressLedger(
            is_request_satisfied=MagenticProgressLedgerItem(reason="Task ongoing.", answer=False),
            is_in_loop=MagenticProgressLedgerItem(reason="No loop detected.", answer=False),
            is_progress_being_made=MagenticProgressLedgerItem(reason="First round.", answer=True),
            next_speaker=MagenticProgressLedgerItem(reason="Only participant.", answer=first_agent),
            instruction_or_question=MagenticProgressLedgerItem(reason="Direct delegation.", answer="Please complete the task."),
        )

    async def prepare_final_answer(self, ctx: MagenticContext) -> Message:
        last = ctx.chat_history[-1] if ctx.chat_history else None
        return last or Message(role="assistant", contents=["Done."])

async def main():
    researcher = Agent(
        client=FoundryChatClient(model="gpt-4o"),
        name="researcher",
        instructions="Research and answer factual questions.",
    )
    manager = SimpleDelegateManager(max_stall_count=2, max_round_count=5)
    workflow = MagenticBuilder(participants=[researcher], manager=manager).build()
    result = await workflow.run("What is the capital of Australia?")
    print(result.text)

asyncio.run(main())
```

**Example 2 — stateful manager with checkpoint save/restore:**

```python
import asyncio
from agent_framework.orchestrations import (
    MagenticManagerBase, MagenticContext,
    MagenticProgressLedger, MagenticProgressLedgerItem,
)
from agent_framework import Message

class StatefulManager(MagenticManagerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._plan_revision_count = 0

    async def plan(self, ctx: MagenticContext) -> Message:
        self._plan_revision_count = 0
        return Message(role="assistant", contents=["Initial plan: gather information."])

    async def replan(self, ctx: MagenticContext) -> Message:
        self._plan_revision_count += 1
        return Message(role="assistant",
                       contents=[f"Revision #{self._plan_revision_count}: refine approach."])

    async def create_progress_ledger(self, ctx: MagenticContext) -> MagenticProgressLedger:
        first_agent = list(ctx.participant_descriptions.keys())[0]
        return MagenticProgressLedger(
            is_request_satisfied=MagenticProgressLedgerItem(reason="Still working.", answer=False),
            is_in_loop=MagenticProgressLedgerItem(reason="No loop detected.", answer=False),
            is_progress_being_made=MagenticProgressLedgerItem(reason=f"Revision #{self._plan_revision_count}.", answer=True),
            next_speaker=MagenticProgressLedgerItem(reason="Only participant.", answer=first_agent),
            instruction_or_question=MagenticProgressLedgerItem(reason="Continuing task.", answer="Continue."),
        )

    async def prepare_final_answer(self, ctx: MagenticContext) -> Message:
        return Message(role="assistant", contents=["Task complete."])

    def on_checkpoint_save(self) -> dict:
        return {"plan_revision_count": self._plan_revision_count}

    def on_checkpoint_restore(self, state: dict) -> None:
        self._plan_revision_count = state.get("plan_revision_count", 0)
```

**Example 3 — limit rounds and resets for a cost-bounded agent:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.orchestrations import (
    MagenticManagerBase, MagenticBuilder,
    MagenticContext, MagenticProgressLedger, MagenticProgressLedgerItem,
)
from agent_framework import Message
from agent_framework.foundry import FoundryChatClient

class BudgetedManager(MagenticManagerBase):
    """Caps the manager at 3 resets and 10 rounds to control costs."""

    async def plan(self, ctx: MagenticContext) -> Message:
        return Message(role="assistant", contents=["Begin research phase."])

    async def replan(self, ctx: MagenticContext) -> Message:
        return Message(role="assistant", contents=["Retrying with a different approach."])

    async def create_progress_ledger(self, ctx: MagenticContext) -> MagenticProgressLedger:
        agents = list(ctx.participant_descriptions.keys())
        return MagenticProgressLedger(
            is_request_satisfied=MagenticProgressLedgerItem(reason="Task ongoing.", answer=False),
            is_in_loop=MagenticProgressLedgerItem(reason="No loop detected.", answer=False),
            is_progress_being_made=MagenticProgressLedgerItem(reason="Working.", answer=True),
            next_speaker=MagenticProgressLedgerItem(reason="First participant.", answer=agents[0] if agents else ""),
            instruction_or_question=MagenticProgressLedgerItem(reason="Continuing.", answer="Continue the task."),
        )

    async def prepare_final_answer(self, ctx: MagenticContext) -> Message:
        return Message(role="assistant", contents=["Final answer prepared."])

async def main():
    worker = Agent(
        client=FoundryChatClient(model="gpt-4o-mini"),
        name="worker",
        instructions="Complete assigned tasks efficiently.",
    )
    manager = BudgetedManager(
        max_stall_count=2,
        max_reset_count=3,
        max_round_count=10,
    )
    workflow = MagenticBuilder(participants=[worker], manager=manager).build()
    result = await workflow.run("Analyse the Q3 sales data.")
    print(result.text)

asyncio.run(main())
```

---

## 8. `BaseGroupChatOrchestrator` and Events

**Module:** `agent_framework.orchestrations`  
**Install:** `pip install agent-framework[orchestrations]`

`BaseGroupChatOrchestrator` is the abstract base class shared by `GroupChatOrchestrator`
(round-robin) and `AgentBasedGroupChatOrchestrator` (LLM-driven selection). It manages
the participant registry, round counting, and termination-condition checking. Subclass it
to build entirely custom group-chat routing strategies.

### Constructor reference

```
BaseGroupChatOrchestrator(
    id: str,
    participant_registry: ParticipantRegistry,
    *,
    name: str | None = None,
    max_rounds: int | None = None,          # None = unlimited; < 1 coerced to 1
    termination_condition: TerminationCondition | None = None,
)
```

### Class constants

| Constant | Value |
|----------|-------|
| `TERMINATION_CONDITION_MET_MESSAGE` | `"The group chat has reached its termination condition."` |
| `MAX_ROUNDS_MET_MESSAGE` | `"The group chat has reached the maximum number of rounds."` |

### Built-in `@handler` methods (all accept `str`, `Message`, or `list[Message]`)

| Handler | Input type | Purpose |
|---------|-----------|---------|
| `handle_str` | `str` | Wraps plain text in a `USER` Message then delegates |
| `handle_message` | `Message` | Wraps a single message in a list then delegates |
| `handle_messages` | `list[Message]` | Validates non-empty then calls `_handle_messages` |
| `handle_participant_response` | `AgentExecutorResponse \| GroupChatResponseMessage` | Receive participant replies; override for custom post-processing |

### Event dataclasses

Both events are emitted via the `RunnerContext` event pipeline and can be observed with
a `ContextProvider.before_run` / `after_run` hook or a middleware.

```python
@dataclass
class GroupChatRequestSentEvent:
    round_index: int
    participant_name: str

@dataclass
class GroupChatResponseReceivedEvent:
    round_index: int
    participant_name: str
```

**Example 1 — observe group chat events via workflow streaming:**

> `GroupChatResponseReceivedEvent` is emitted as a `WorkflowEvent` with
> `type="group_chat"`. Use `workflow.run(stream=True)` to observe them — the event
> is not accessible through `SessionContext` fields inside a `ContextProvider`.

```python
import asyncio
from agent_framework import Agent, WorkflowEvent
from agent_framework.orchestrations import (
    GroupChatBuilder, GroupChatState, GroupChatResponseReceivedEvent,
)
from agent_framework.foundry import FoundryChatClient

async def main():
    client = FoundryChatClient(model="gpt-4o")
    writer = Agent(client=client, name="writer", instructions="Write clearly.")
    reviewer = Agent(client=client, name="reviewer", instructions="Review and improve.")

    def round_robin(state: GroupChatState) -> str:
        names = list(state.participants.keys())
        return names[state.current_round % len(names)]

    workflow = GroupChatBuilder(
        participants=[writer, reviewer],
        selection_func=round_robin,
        max_rounds=4,
    ).build()

    # stream=True yields WorkflowEvent objects; group-chat events have type="group_chat"
    stream = workflow.run("Write a tweet about renewable energy.", stream=True)
    async for event in stream:
        if event.type == "group_chat" and isinstance(event.data, GroupChatResponseReceivedEvent):
            print(f"[Round {event.data.round_index}] ← {event.data.participant_name}")
    result = await stream.get_final_response()
    print(result.text)

asyncio.run(main())
```

**Example 2 — keyword routing via `selection_func`:**

> **Subclassing `BaseGroupChatOrchestrator`** requires implementing `_handle_messages` and
> `_handle_response` (both raise `NotImplementedError` in the base class). For most custom
> routing needs, `selection_func=` is the recommended approach — it receives a
> `GroupChatState` snapshot and returns the name of the next participant.

```python
import asyncio
from agent_framework import Agent
from agent_framework.orchestrations import GroupChatBuilder, GroupChatState
from agent_framework.foundry import FoundryChatClient

ROUTING = {
    "code": "coder",
    "test": "tester",
    "deploy": "devops",
}

def keyword_router(state: GroupChatState) -> str:
    """Route to the agent whose keyword appears in the latest message."""
    if state.conversation:
        for content in reversed(state.conversation[-1].contents):
            if isinstance(content, str):
                text = content.lower()
                for keyword, agent_name in ROUTING.items():
                    if keyword in text and agent_name in state.participants:
                        return agent_name
    # Default: first participant
    return next(iter(state.participants))

async def main():
    client = FoundryChatClient(model="gpt-4o")
    coder = Agent(client=client, name="coder", instructions="Write code.")
    tester = Agent(client=client, name="tester", instructions="Write tests.")
    devops = Agent(client=client, name="devops", instructions="Handle deployment.")

    workflow = GroupChatBuilder(
        participants=[coder, tester, devops],
        selection_func=keyword_router,
        max_rounds=3,
    ).build()
    result = await workflow.run("Please write tests for the login module.")
    print(result.text)

asyncio.run(main())
```

**Example 3 — cap rounds to prevent runaway conversations:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.orchestrations import GroupChatBuilder, TerminationCondition
from agent_framework import Message
from agent_framework.foundry import FoundryChatClient

def stop_on_consensus(messages: list[Message]) -> bool:
    """Stop the chat when the last message contains 'AGREED'."""
    if not messages:
        return False
    last = messages[-1]
    for content in last.contents:
        if isinstance(content, str) and "AGREED" in content.upper():
            return True
    return False

async def main():
    client = FoundryChatClient(model="gpt-4o")
    agent_a = Agent(client=client, name="agent_a", instructions="Propose a solution.")
    agent_b = Agent(client=client, name="agent_b", instructions="Critique the solution.")

    def alternate(state):
        names = list(state.participants.keys())
        return names[state.current_round % len(names)]

    workflow = GroupChatBuilder(
        participants=[agent_a, agent_b],
        selection_func=alternate,
        max_rounds=6,
        termination_condition=stop_on_consensus,
    ).build()
    result = await workflow.run("Should we migrate to microservices?")
    print(result.text)

asyncio.run(main())
```

---

## 9. `AgentRequestInfoResponse` + `CacheProvider`

**Module:** `agent_framework.orchestrations` (AgentRequestInfoResponse), `agent_framework.microsoft` (CacheProvider)

### `AgentRequestInfoResponse`

A dataclass used in **HITL orchestration flows** to return additional user-supplied
context to an agent that raised a `request_info` interrupt. The framework sends the
`messages` list back to the waiting agent and resumes execution.

```python
@dataclass
class AgentRequestInfoResponse:
    messages: list[Message]
```

#### Factory methods (source-verified)

| Method | Signature | Behaviour |
|--------|-----------|-----------|
| `from_messages(messages)` | `list[Message] -> AgentRequestInfoResponse` | Direct wrapping of pre-built `Message` objects |
| `from_strings(texts)` | `list[str] -> AgentRequestInfoResponse` | Each string becomes a `Message(role="user", contents=[text])` |
| `approve()` | `() -> AgentRequestInfoResponse` | Empty `messages` list — signals that the original agent response is accepted as-is |

**Key distinction:** `approve()` is the right response when a human reviews an agent's
pending response and chooses to let it through unchanged. `from_strings(["..."])` is used
when the human wants to add or correct information before the agent continues.

### `CacheProvider`

A structural `Protocol` that defines a simple async cache interface consumed by the
Purview integration (and any code that requires an injectable cache).

```python
class CacheProvider(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None: ...
    async def remove(self, key: str) -> None: ...
```

There is no built-in implementation in `agent_framework.microsoft` — the most common
pattern is to provide an in-memory dict-backed implementation for tests and a Redis-backed
one in production.

**Example 1 — `AgentRequestInfoResponse` factory methods:**

> `AgentRequestInfoResponse` is the response object you send back to a paused workflow.
> In a HITL flow the workflow emits a `request_info` event and suspends; you resume it
> with `workflow.run(responses={event.request_id: AgentRequestInfoResponse(...)})`.
> The `@response_handler` decorator belongs on executor classes, not on `Agent`.

```python
from agent_framework.orchestrations import AgentRequestInfoResponse
from agent_framework import Message

# 1. Approve — continue with the agent's existing draft
approval = AgentRequestInfoResponse.approve()
print(f"Approve payload messages: {len(approval.messages)}")  # 0

# 2. Provide corrections as plain strings (converted to Messages internally)
correction = AgentRequestInfoResponse.from_strings([
    "Please use a more empathetic tone.",
    "Mention the estimated recovery time of 2 hours.",
])
print(f"Correction messages: {len(correction.messages)}")  # 2

# 3. Provide fully structured Message objects
structured = AgentRequestInfoResponse.from_messages([
    Message(role="user", contents=["The outage lasted 2 hours."]),
    Message(role="user", contents=["Affected 1,200 customers in the EU region."]),
])
print(f"Structured messages: {len(structured.messages)}")  # 2

# In a workflow HITL loop:
# events = result.get_request_info_events()
# resumed = await workflow.run(responses={events[0].request_id: correction})
```

**Example 2 — pass structured `Message` objects back from a HITL step:**

```python
import asyncio
from agent_framework import Agent, Message
from agent_framework.orchestrations import AgentRequestInfoResponse
from agent_framework.foundry import FoundryChatClient

async def main():
    agent = Agent(
        client=FoundryChatClient(model="gpt-4o"),
        instructions="Ask for clarification when needed.",
    )
    # Simulate injecting context from an external system
    response = AgentRequestInfoResponse.from_messages([
        Message(role="user", contents=["The budget is $50,000 and deadline is Q2."]),
        Message(role="user", contents=["Stakeholder is Alice in Finance."]),
    ])
    print(f"Injecting {len(response.messages)} message(s) into the agent context.")

asyncio.run(main())
```

**Example 3 — in-memory `CacheProvider` implementation for tests:**

```python
import asyncio
from typing import Any
from agent_framework.microsoft import CacheProvider

class InMemoryCache:
    """Simple in-memory CacheProvider implementation for testing."""

    def __init__(self):
        self._store: dict[str, Any] = {}

    async def get(self, key: str) -> Any | None:
        return self._store.get(key)

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        self._store[key] = value  # ttl ignored in this simple impl

    async def remove(self, key: str) -> None:
        self._store.pop(key, None)

# Verify it satisfies the protocol
cache: CacheProvider = InMemoryCache()  # structural typing check

async def main():
    await cache.set("policy-result:user-42:turn-1", {"allowed": True}, ttl_seconds=300)
    result = await cache.get("policy-result:user-42:turn-1")
    print(f"Cached result: {result}")
    await cache.remove("policy-result:user-42:turn-1")
    print(f"After remove: {await cache.get('policy-result:user-42:turn-1')}")

asyncio.run(main())
```

---

## 10. Purview exception hierarchy + `acquire_token`

**Module:** `agent_framework.microsoft`  
**Install:** `pip install agent-framework[purview]`

### Exception hierarchy

All Purview-related exceptions follow a two-level hierarchy:

```
AgentFrameworkException
  └── IntegrationException                  (agent_framework)
        ├── IntegrationInvalidAuthException
        │     └── PurviewAuthenticationError   # 401/403
        └── PurviewServiceError              # base for HTTP errors
              ├── PurviewPaymentRequiredError  # 402
              ├── PurviewRateLimitError        # 429
              └── PurviewRequestError          # other non-2xx HTTP errors
```

| Exception | HTTP status | When raised |
|-----------|-------------|-------------|
| `PurviewAuthenticationError` | 401 / 403 | Invalid or expired credentials; insufficient permission |
| `PurviewPaymentRequiredError` | 402 | Subscription or quota exhausted |
| `PurviewRateLimitError` | 429 | Throttled by the Purview service |
| `PurviewRequestError` | other 4xx/5xx | Any other non-success HTTP response |
| `PurviewServiceError` | (base) | Catch all Purview HTTP errors without subtyping |

### `acquire_token`

```
acquire_token(
    *,
    client_id: str,      # AAD application (client) ID
    tenant_id: str,      # AAD tenant ID
    username: str | None = None,       # optional account hint for silent acquisition
    token_cache: Any | None = None,    # MSAL token cache instance
    scopes: list[str] | None = None,   # defaults to Power Platform API scopes
) -> str                               # access token string
```

`acquire_token` implements a **silent-then-interactive** MSAL token acquisition flow:

1. Build a `PublicClientApplication` with the given authority.
2. Attempt silent acquisition using any cached accounts (`get_accounts()`).
3. If silent acquisition fails (no cached token, expired token), fall back to interactive browser-based acquisition.
4. Raise `AgentException` if both paths fail.

**Example 1 — catch and handle the full Purview exception hierarchy:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.microsoft import (
    PurviewPolicyMiddleware,
    PurviewAuthenticationError,
    PurviewRateLimitError,
    PurviewPaymentRequiredError,
    PurviewRequestError,
    PurviewServiceError,
    PurviewSettings,
)
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential

async def main():
    settings = PurviewSettings(
        app_name="my-agent-app",
        tenant_id="my-tenant-id",
    )
    # PurviewPolicyMiddleware requires (credential, settings) — credential first
    credential = DefaultAzureCredential()
    purview_mw = PurviewPolicyMiddleware(credential, settings)
    agent = Agent(
        client=FoundryChatClient(model="gpt-4o"),
        middleware=[purview_mw],
    )
    try:
        result = await agent.run("Classify this document.")
        print(result.text)
    except PurviewAuthenticationError:
        print("ERROR: credential rejected — rotate the service principal secret.")
    except PurviewRateLimitError:
        print("ERROR: rate limited — implement exponential back-off.")
    except PurviewPaymentRequiredError:
        print("ERROR: subscription quota exceeded — check the Azure portal.")
    except PurviewRequestError as e:
        print(f"ERROR: Purview HTTP error: {e}")
    except PurviewServiceError as e:
        print(f"ERROR: unclassified Purview error: {e}")

asyncio.run(main())
```

**Example 2 — acquire a Power Platform token silently with a cached account:**

```python
from msal import SerializableTokenCache
from agent_framework.microsoft import acquire_token

def get_token(client_id: str, tenant_id: str, cache_path: str = "/tmp/msal_cache.json") -> str:
    """Acquire a token with on-disk MSAL cache for CI/CD pipelines."""
    cache = SerializableTokenCache()
    try:
        with open(cache_path) as f:
            cache.deserialize(f.read())
    except FileNotFoundError:
        pass  # no cache yet — first run will use interactive auth

    token = acquire_token(
        client_id=client_id,
        tenant_id=tenant_id,
        token_cache=cache,
    )

    if cache.has_state_changed:
        with open(cache_path, "w") as f:
            f.write(cache.serialize())

    return token

# Usage
token = get_token("your-client-id", "your-tenant-id")
print(f"Token acquired ({len(token)} chars)")
```

**Example 3 — custom scopes for non-Power-Platform APIs + retry with back-off:**

> **Note:** `acquire_token()` raises `AgentException` on all failures (MSAL silent/interactive
> failures and no-token outcomes). It does **not** raise `PurviewRateLimitError` — that
> exception is raised by the Purview HTTP middleware when the Purview API returns a 429, not
> by `acquire_token`. Retry logic around `acquire_token` should therefore catch `AgentException`.

```python
import time
from agent_framework.exceptions import AgentException
from agent_framework.microsoft import acquire_token

AZURE_MANAGEMENT_SCOPES = ["https://management.azure.com/.default"]

def acquire_with_retry(client_id: str, tenant_id: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            return acquire_token(
                client_id=client_id,
                tenant_id=tenant_id,
                scopes=AZURE_MANAGEMENT_SCOPES,
            )
        except AgentException:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # exponential back-off: 1s, 2s, 4s
                print(f"Token acquisition failed. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Token acquisition failed after max retries")

token = acquire_with_retry("your-client-id", "your-tenant-id")
print("Token ready")
```

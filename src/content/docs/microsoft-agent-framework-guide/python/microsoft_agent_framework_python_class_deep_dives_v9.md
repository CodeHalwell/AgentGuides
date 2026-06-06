---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 9"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.8.0: OllamaChatClient, PurviewPolicyMiddleware, DurableAIAgent+Worker+Client, GitHubCopilotAgent, HyperlightExecuteCodeTool, HyperlightCodeActProvider, Mem0ContextProvider, RedisContextProvider+RedisHistoryProvider, StandardMagenticManager+MagenticContext, FileSkillsSource+FilteringSkillsSource."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 32
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 9

Verified against **agent-framework 1.8.0** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source. No API name has been guessed or inferred from documentation alone.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, `AgentMiddleware`, `ChatMiddleware`, `FunctionMiddleware`, `CompactionProvider`, `ToolResultCompactionStrategy`, `TokenBudgetComposedStrategy`, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — `BackgroundAgentsProvider`, `MemoryContextProvider`, `TodoProvider`, `AgentModeProvider`, `SummarizationStrategy`, `ContextWindowCompactionStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`, `WorkflowViz`, `MCPStreamableHTTPTool` + `MCPWebsocketTool`
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — `Message` + `Content`, `ChatOptions` + `ChatResponse`, `ResponseStream`, `AgentContext`, `FunctionalWorkflow` + `StepWrapper`, `WorkflowEvent` taxonomy, `SkillsSource` composition, `EvalItem` + `EvalResults`, `TokenizerProtocol`, `ConversationSplit`
- [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) — `Executor` + `@handler` + `@executor`, `AgentExecutor`, edge groups, `Runner`, `SessionContext`, `AgentSession`, `BaseChatClient`, `SecretString`, `WorkflowCheckpoint`, exception hierarchy
- [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) — `ExperimentalFeature`, `WorkflowRunState`, `WorkflowExecutor`, `AgentResponse`, `BaseEmbeddingClient`, `FunctionInvocationConfiguration`, `ClassSkill`, `Annotation`, capability protocols, middleware layers
- [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) — `ContextProvider`, `BackgroundTaskInfo`, `GroupChatBuilder`, `HandoffBuilder`, `MagenticBuilder`, `SequentialBuilder`, `ConcurrentBuilder`, `AgentFactory`, `WorkflowFactory`, `SecureAgentConfig`, `FunctionalWorkflowAgent`, `ObservabilitySettings`
- [Vol. 8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) — `AgentFileStore` hierarchy, `FileAccessProvider`, `MCPSkill` + `MCPSkillsSource`, `ToolMode`, `AgentEvalConverter` + `CheckResult` + `RubricScore`, `ChatContext`, `WorkflowAgent` + `WorkflowContext`, `TruncationStrategy`, `HistoryProvider` + `InMemoryHistoryProvider`, `DelegatingSkillsSource` + `InMemorySkillsSource` + `FunctionInvocationContext`

This volume covers **ten new class groups** drawn from the optional integration
packages that ship alongside `agent-framework 1.8.0`: the **local-LLM bridge**
(`OllamaChatClient`), **enterprise content compliance** (`PurviewPolicyMiddleware`),
**durable workflow actors** (`DurableAIAgent` / `DurableAIAgentClient` / `DurableAIAgentWorker`),
**GitHub Copilot routing** (`GitHubCopilotAgent`), **Hyperlight sandboxed code execution**
(`HyperlightExecuteCodeTool` + `HyperlightCodeActProvider`), **cloud semantic memory**
(`Mem0ContextProvider`), **Redis persistence** (`RedisContextProvider` + `RedisHistoryProvider`),
**Magentic-One orchestration internals** (`StandardMagenticManager` + `MagenticContext` + `MagenticProgressLedger`),
and **filesystem skill discovery** (`FileSkillsSource` + `FilteringSkillsSource` + `AggregatingSkillsSource`).

---

## Table of Contents

1. [`OllamaChatClient` + `OllamaChatOptions` + `OllamaSettings`](#1-ollamachatclient--ollamachatoptions--ollamasettings)
2. [`PurviewPolicyMiddleware` + `PurviewChatPolicyMiddleware` + `PurviewSettings` + `PurviewAppLocation`](#2-purviewpolicymiddleware--purviewchatpolicymiddleware--purviewsettings--purviewapplocation)
3. [`DurableAIAgent` + `DurableAIAgentClient` + `DurableAIAgentWorker`](#3-durableaiagent--durableaiagentclient--durableaiagentworker)
4. [`GitHubCopilotAgent` + `GitHubCopilotOptions` + `GitHubCopilotSettings`](#4-githubcopilotagent--githubcopyiloptions--githubcopilotsettings)
5. [`HyperlightExecuteCodeTool` + `AllowedDomain` + `FileMount`](#5-hyperlightexecutecodetool--alloweddomain--filemount)
6. [`HyperlightCodeActProvider`](#6-hyperlightcodeactprovider)
7. [`Mem0ContextProvider`](#7-mem0contextprovider)
8. [`RedisContextProvider` + `RedisHistoryProvider`](#8-rediscontextprovider--redishistoryprovider)
9. [`StandardMagenticManager` + `MagenticContext` + `MagenticProgressLedger`](#9-standardmagenticmanager--magenticcontext--magenticprogressledger)
10. [`FileSkillsSource` + `FileSkill` + `FilteringSkillsSource` + `AggregatingSkillsSource`](#10-fileskillssource--fileskill--filteringskillssource--aggregatingskillssource)

---

## 1. `OllamaChatClient` + `OllamaChatOptions` + `OllamaSettings`

**Source:** `agent_framework_ollama._chat_client`  
**Package:** `pip install agent-framework[ollama]`

`OllamaChatClient` is a full-stack chat client for locally-hosted [Ollama](https://ollama.com)
models. It implements `FunctionInvocationLayer`, `ChatMiddlewareLayer`, and `ChatTelemetryLayer`
so you get tool calling, middleware chaining, and OpenTelemetry tracing out of the box —
exactly the same surface as `OpenAIChatClient` or `FoundryChatClient`.

### Constructor signature

```python
class OllamaChatClient(BaseChatClient):
    def __init__(
        self,
        *,
        host: str | None = None,          # Ollama URL; env OLLAMA_HOST; default "http://localhost:11434"
        client: AsyncClient | None = None, # Pre-built ollama.AsyncClient
        model: str | None = None,          # env OLLAMA_MODEL (required)
        additional_properties: dict[str, Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...
```

### Key `OllamaChatOptions` fields

| Key | Type | Maps to Ollama field | Notes |
|-----|------|----------------------|-------|
| `model` | `str` | `model` | Override per call |
| `temperature` | `float` | `options.temperature` | Sampling temp |
| `max_tokens` | `int` | `options.num_predict` | Max generated tokens |
| `top_p` | `float` | `options.top_p` | Nucleus sampling |
| `seed` | `int` | `options.seed` | Reproducible output |
| `top_k` | `int` | `options.top_k` | Limit to top-k tokens |
| `num_ctx` | `int` | `options.num_ctx` | Context window size |
| `repeat_penalty` | `float` | `options.repeat_penalty` | Penalise token repetition |
| `keep_alive` | `str \| int` | `keep_alive` | How long to keep model loaded (`"5m"`, `0`, …) |
| `think` | `bool` | `think` | Enable reasoning chain for thinking models |
| `response_format` | `str \| dict` | `format` | `"json"` or JSON schema dict |

`tool_choice`, `user`, `store`, `logit_bias`, and `metadata` are typed `None`
(Ollama does not support them).

### Example 1 — basic local agent

```python
import asyncio
import os
from agent_framework import Agent, tool
from agent_framework_ollama import OllamaChatClient

@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32

async def main():
    client = OllamaChatClient(model="llama3.2")
    agent = Agent(
        client=client,
        instructions="You are a helpful unit-conversion assistant.",
        tools=[celsius_to_fahrenheit],
    )
    response = await agent.run("What is 37°C in Fahrenheit?")
    print(response.text)

asyncio.run(main())
```

### Example 2 — structured JSON output

```python
from typing import TypedDict
import asyncio
from agent_framework_ollama import OllamaChatClient

class SentimentResult(TypedDict):
    label: str       # "positive" | "negative" | "neutral"
    score: float     # 0.0 – 1.0

schema = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["label", "score"],
}

async def classify(text: str) -> SentimentResult:
    client = OllamaChatClient(model="llama3.2")
    from agent_framework._types import Message
    response = await client.get_response(
        messages=[Message(role="user", contents=[f"Classify sentiment: {text}"])],
        options={"response_format": schema},
    )
    import json
    return json.loads(response.messages[-1].text)  # type: ignore[index]

asyncio.run(classify("I absolutely love this framework!"))
```

### Example 3 — streaming with keep_alive control

```python
import asyncio
from agent_framework import Agent
from agent_framework_ollama import OllamaChatClient, OllamaChatOptions

async def main():
    # Set keep_alive=0 to unload model immediately after the call
    client = OllamaChatClient(model="mistral")
    agent = Agent(client=client, instructions="You are a concise assistant.")

    options: OllamaChatOptions = {
        "keep_alive": 0,       # release GPU memory immediately
        "temperature": 0.3,
        "num_ctx": 2048,
    }
    async for chunk in await agent.run("Summarise the Turing test in one sentence.",
                                       stream=True,
                                       options=options):
        print(chunk.text, end="", flush=True)
    print()

asyncio.run(main())
```

### Example 4 — multi-turn session with Ollama

```python
import asyncio
from agent_framework import Agent
from agent_framework_ollama import OllamaChatClient

async def main():
    client = OllamaChatClient(model="llama3.2")
    agent = Agent(client=client, instructions="You are a helpful math tutor.")
    session = agent.create_session()

    turns = [
        "What is the quadratic formula?",
        "Can you show me an example using x² + 5x + 6?",
        "What are the roots?",
    ]
    for prompt in turns:
        response = await agent.run(prompt, session=session)
        print(f"User: {prompt}")
        print(f"Agent: {response.text}\n")

asyncio.run(main())
```

---

## 2. `PurviewPolicyMiddleware` + `PurviewChatPolicyMiddleware` + `PurviewSettings` + `PurviewAppLocation`

**Source:** `agent_framework_purview._middleware`, `_settings`  
**Package:** `pip install agent-framework[purview]`

These classes integrate **Microsoft Purview** content-policy enforcement into the framework's
middleware pipeline. Both variants evaluate outgoing messages (upload) and incoming responses
(download) against your organisation's Purview data-loss-prevention policies:

| Class | Layer | Use when… |
|-------|-------|-----------|
| `PurviewPolicyMiddleware` | `AgentMiddleware` | You want policy applied at the agent level, with access to the full `AgentContext` (session ID auto-resolved from the session or `conversation_id` property) |
| `PurviewChatPolicyMiddleware` | `ChatMiddleware` | You want policy applied to a specific chat client regardless of which agent uses it |

### Constructor — `PurviewPolicyMiddleware`

```python
class PurviewPolicyMiddleware(AgentMiddleware):
    def __init__(
        self,
        credential: AzureCredentialTypes | AzureTokenProvider,
        settings: PurviewSettings,
        cache_provider: CacheProvider | None = None,
    ) -> None: ...
```

### `PurviewSettings` fields

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `app_name` | `str \| None` | — | Identifier for your application in Purview logs |
| `app_version` | `str \| None` | — | Application version string |
| `tenant_id` | `str \| None` | — | Azure tenant GUID |
| `purview_app_location` | `PurviewAppLocation \| None` | — | Registered app location for policy routing |
| `graph_base_uri` | `str \| None` | — | Override Microsoft Graph base URI |
| `blocked_prompt_message` | `str \| None` | `"Prompt blocked by policy"` | User-visible message when input is blocked |
| `blocked_response_message` | `str \| None` | `"Response blocked by policy"` | User-visible message when response is blocked |
| `ignore_exceptions` | `bool \| None` | `False` | Log but swallow all Purview errors |
| `ignore_payment_required` | `bool \| None` | `False` | Log but swallow HTTP 402 errors |
| `cache_ttl_seconds` | `int \| None` | 14400 (4 h) | Policy decision cache TTL |
| `max_cache_size_bytes` | `int \| None` | 200 MB | Maximum in-process cache size |

`PurviewAppLocation` wraps a `PurviewLocationType` enum (`APPLICATION`, `URI`, `DOMAIN`)
plus a string `location_value`, and serialises to the `@odata.type` payload the Graph API expects.

### Example 1 — agent-level compliance enforcement

```python
import asyncio
from azure.identity import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_purview import PurviewPolicyMiddleware, PurviewSettings

async def main():
    settings: PurviewSettings = {
        "app_name": "CustomerSupportBot",
        "app_version": "2.1.0",
        "blocked_prompt_message": "Your message was flagged by our content policy. Please rephrase.",
        "blocked_response_message": "The response was redacted by our content policy.",
        "ignore_exceptions": False,   # fail hard on Purview errors
    }

    credential = DefaultAzureCredential()
    agent = Agent(
        client=FoundryChatClient(),
        instructions="You are a customer support agent.",
        middleware=[PurviewPolicyMiddleware(credential, settings)],
    )

    response = await agent.run("How do I reset my password?")
    print(response.text)

asyncio.run(main())
```

### Example 2 — chat-client level policy (shared across agents)

```python
import asyncio
from azure.identity import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_purview import PurviewChatPolicyMiddleware, PurviewSettings

settings: PurviewSettings = {
    "app_name": "EnterpriseAgent",
    "ignore_payment_required": True,  # Purview trial has limited capacity
}

credential = DefaultAzureCredential()
purview_middleware = PurviewChatPolicyMiddleware(credential, settings)

# Both agents share the same policy-enforced chat client
client = FoundryChatClient(middleware=[purview_middleware])

agent_a = Agent(client=client, instructions="Answer questions about HR policy.")
agent_b = Agent(client=client, instructions="Answer questions about IT infrastructure.")
```

### Example 3 — app location for fine-grained policy routing

```python
from agent_framework_purview import PurviewSettings, PurviewAppLocation, PurviewLocationType

# Route policy evaluation to a specific registered URI
settings: PurviewSettings = {
    "app_name": "LegalReviewBot",
    "purview_app_location": PurviewAppLocation(
        location_type=PurviewLocationType.URI,
        location_value="https://legalbot.contoso.com/chat",
    ),
}
# PurviewAppLocation.get_policy_location() returns the Graph API payload:
# {"@odata.type": "microsoft.graph.policyLocationUrl", "value": "https://..."}
loc = settings["purview_app_location"]
print(loc.get_policy_location())
```

### Example 4 — fault-tolerant mode for CI/staging environments

```python
import asyncio
from azure.identity import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from agent_framework_purview import PurviewPolicyMiddleware, PurviewSettings

async def build_agent(enforce: bool) -> Agent:
    settings: PurviewSettings = {
        "app_name": "MyApp",
        "ignore_exceptions": not enforce,       # soft-fail in non-prod
        "ignore_payment_required": not enforce,
    }
    middleware = [PurviewPolicyMiddleware(DefaultAzureCredential(), settings)]
    return Agent(
        client=OpenAIChatCompletionClient(model="gpt-4.1"),
        instructions="You are a helpful assistant.",
        middleware=middleware,
    )

async def main():
    import os
    agent = await build_agent(enforce=os.getenv("ENV") == "production")
    response = await agent.run("Draft a contract clause for SLA 99.9%.")
    print(response.text)

asyncio.run(main())
```

---

## 3. `DurableAIAgent` + `DurableAIAgentClient` + `DurableAIAgentWorker`

**Source:** `agent_framework_durabletask._shim`, `_client`, `_worker`  
**Package:** `pip install agent-framework[durabletask]`

These three classes expose agents as **Durable Task Framework entities** — persistent, reliable
actors that survive process restarts. The trio covers the full lifecycle:

| Class | Role |
|-------|------|
| `DurableAIAgentWorker` | Registers `Agent` instances as durable entities on a `TaskHubGrpcWorker` |
| `DurableAIAgentClient` | External caller: retrieves agent proxies from a `TaskHubGrpcClient` and invokes them synchronously |
| `DurableAIAgent` | Proxy shim returned by the client/orchestration context; has the same `.run()` surface as `Agent` but returns a `Task` instead of a coroutine |

### Signatures

```python
class DurableAIAgentWorker:
    def __init__(
        self,
        worker: TaskHubGrpcWorker,
        callback: AgentResponseCallbackProtocol | None = None,
    ) -> None: ...

    def add_agent(
        self,
        agent: SupportsAgentRun,
        callback: AgentResponseCallbackProtocol | None = None,
    ) -> None: ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    registered_agent_names: list[str]        # property

class DurableAIAgentClient:
    def __init__(
        self,
        client: TaskHubGrpcClient,
        max_poll_retries: int = 30,
        poll_interval_seconds: float = 2.0,
    ) -> None: ...

    def get_agent(self, agent_name: str) -> DurableAIAgent[AgentResponse]: ...

class DurableAIAgent(Generic[TaskT]):
    id: str
    name: str
    display_name: str
    description: str | None

    def __init__(
        self,
        executor: DurableAgentExecutor[TaskT],
        name: str,
        *,
        agent_id: str | None = None,
    ) -> None: ...

    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: Literal[False] = False,
        session: AgentSession | None = None,
        options: dict[str, Any] | None = None,
    ) -> TaskT: ...                          # sync Task — yield in orchestration context

    def create_session(self, *, session_id: str | None = None) -> DurableAgentSession: ...
    def get_session(self, service_session_id: str, *, session_id: str | None = None) -> AgentSession: ...
```

> **Important:** `DurableAIAgent.run()` is **synchronous** and returns a durable `Task` object —
> not a coroutine. Inside a Durable orchestration you `yield` it. Outside an orchestration
> (via `DurableAIAgentClient`) it blocks until the entity responds.

### Example 1 — register and start a worker

```python
# worker.py
from durabletask.worker import TaskHubGrpcWorker
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_durabletask import DurableAIAgentWorker

worker = TaskHubGrpcWorker(host_address="localhost:4001")
agent_worker = DurableAIAgentWorker(worker)

# Register multiple agents as separate entities
client = FoundryChatClient()
summariser = Agent(client=client, name="summariser",
                   instructions="Summarise the provided text concisely.")
researcher = Agent(client=client, name="researcher",
                   instructions="Research the topic and provide key facts.")

agent_worker.add_agent(summariser)
agent_worker.add_agent(researcher)

print("Registered agents:", agent_worker.registered_agent_names)
# → ['summariser', 'researcher']

agent_worker.start()   # blocks; Ctrl-C to stop
```

### Example 2 — invoke from an external client

```python
# client.py
from durabletask import TaskHubGrpcClient
from agent_framework.azure import DurableAIAgentClient

grpc_client = TaskHubGrpcClient(host_address="localhost:4001")
agent_client = DurableAIAgentClient(grpc_client, poll_interval_seconds=1.0)

agent = agent_client.get_agent("summariser")
response = agent.run("The Apollo programme landed humans on the Moon six times between 1969 and 1972.")
print(response.text)
```

### Example 3 — orchestrating agents inside a Durable orchestration

```python
# orchestration.py
import durabletask.task as dt
from agent_framework_durabletask import DurableAIAgentWorker

def research_and_summarise(ctx: dt.OrchestrationContext, topic: str) -> str:
    # ctx.call_entity returns a Task — yield to schedule it
    from agent_framework_durabletask import DurableAIAgent  # type hint only

    researcher_result = yield ctx.call_entity(
        entity_id=dt.EntityInstanceId("dafx-researcher", "singleton"),
        operation="run",
        input={"message": f"Find key facts about: {topic}"},
    )
    facts = researcher_result["text"]

    summariser_result = yield ctx.call_entity(
        entity_id=dt.EntityInstanceId("dafx-summariser", "singleton"),
        operation="run",
        input={"message": f"Summarise these facts:\n{facts}"},
    )
    return summariser_result["text"]
```

### Example 4 — per-agent response callback

```python
from agent_framework_durabletask import DurableAIAgentWorker
from agent_framework_durabletask._callbacks import AgentResponseCallbackProtocol, AgentCallbackContext
from durabletask.worker import TaskHubGrpcWorker
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient

class AuditLogger(AgentResponseCallbackProtocol):
    async def on_response(self, context: AgentCallbackContext) -> None:
        print(f"[AUDIT] agent={context.agent_name} "
              f"tokens={context.response.usage_details}")

worker = TaskHubGrpcWorker(host_address="localhost:4001")
agent_worker = DurableAIAgentWorker(worker, callback=AuditLogger())

my_agent = Agent(client=FoundryChatClient(), name="assistant")
agent_worker.add_agent(my_agent)
agent_worker.start()
```

---

## 4. `GitHubCopilotAgent` + `GitHubCopilotOptions` + `GitHubCopilotSettings`

**Source:** `agent_framework_github_copilot._agent`  
**Package:** `pip install agent-framework[github-copilot]`

`GitHubCopilotAgent` routes agent calls through the **GitHub Copilot CLI** (`gh copilot`),
giving you access to Copilot's model routing (GPT-4.1, Claude Sonnet, Gemini Pro, …) without
managing API keys. It implements the full middleware + telemetry stack.

### Constructor signature

```python
class GitHubCopilotAgent(AgentMiddlewareLayer, AgentTelemetryLayer, RawGitHubCopilotAgent):
    def __init__(
        self,
        instructions: str | None = None,
        *,
        client: CopilotClient | None = None,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        context_providers: Sequence[ContextProvider] | None = None,
        middleware: Sequence[AgentMiddlewareTypes] | None = None,
        tools: ToolTypes | Callable | Sequence[ToolTypes | Callable] | None = None,
        default_options: GitHubCopilotOptions | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...
```

### `GitHubCopilotOptions` fields

| Key | Type | Env variable | Description |
|-----|------|-------------|-------------|
| `model` | `str` | `GITHUB_COPILOT_MODEL` | Model name, e.g. `"gpt-4.1"`, `"claude-sonnet-4-5"` |
| `cli_path` | `str` | `GITHUB_COPILOT_CLI_PATH` | Absolute path to the Copilot CLI binary |
| `timeout` | `float` | `GITHUB_COPILOT_TIMEOUT` | Per-call timeout in seconds (default 60) |
| `log_level` | `str` | `GITHUB_COPILOT_LOG_LEVEL` | CLI log verbosity (`"debug"`, `"info"`, …) |
| `system_message` | `SystemMessageConfig` | — | Append or replace the system prompt |
| `mcp_servers` | `dict[str, MCPServerConfig]` | — | MCP server configurations (stdio or HTTP) |
| `provider` | `ProviderConfig` | — | BYOK provider config (OpenAI, Azure, Anthropic) |
| `instruction_directories` | `list[str]` | — | Additional custom instruction directories |
| `on_permission_request` | `PermissionHandlerType` | — | Callback for CLI permission prompts |
| `on_function_approval` | `FunctionApprovalCallback` | — | Callback gating `approval_mode="always_require"` tools |

### Example 1 — minimal usage (CLI in PATH)

```python
import asyncio
from agent_framework_github_copilot import GitHubCopilotAgent

async def main():
    async with GitHubCopilotAgent(
        instructions="You are a helpful Python code review assistant."
    ) as agent:
        response = await agent.run("Review this code: `x = lambda f: f(f)`")
        print(response.text)

asyncio.run(main())
```

### Example 2 — pin a specific Copilot model

```python
import asyncio
from agent_framework_github_copilot import GitHubCopilotAgent, GitHubCopilotOptions

async def main():
    options: GitHubCopilotOptions = {
        "model": "claude-sonnet-4-5",
        "timeout": 120,
    }
    async with GitHubCopilotAgent(
        instructions="You are a senior software architect.",
        default_options=options,
    ) as agent:
        response = await agent.run("Design a CQRS architecture for an e-commerce platform.")
        print(response.text)

asyncio.run(main())
```

### Example 3 — tools with approval gating

```python
import asyncio
from agent_framework import tool
from agent_framework_github_copilot import GitHubCopilotAgent

@tool(approval_mode="always_require")
def delete_file(path: str) -> str:
    """Delete a file. Requires explicit human approval."""
    import os
    os.remove(path)
    return f"Deleted {path}"

async def approve(tool_call, context):
    # Show the user what's about to happen and get confirmation
    answer = input(f"Allow {tool_call.name}({tool_call.arguments})? [y/N] ")
    return answer.strip().lower() == "y"

async def main():
    async with GitHubCopilotAgent(
        instructions="You are a file management assistant.",
        tools=[delete_file],
        default_options={"on_function_approval": approve},
    ) as agent:
        response = await agent.run("Clean up all .tmp files in /tmp/workdir.")
        print(response.text)

asyncio.run(main())
```

### Example 4 — BYOK via custom provider

```python
import asyncio
from agent_framework_github_copilot import GitHubCopilotAgent, GitHubCopilotOptions

async def main():
    options: GitHubCopilotOptions = {
        "provider": {
            "type": "azure",
            "endpoint": "https://myazure.openai.azure.com/",
            "deployment": "gpt-4o",
            "api_key": "...",
        },
    }
    async with GitHubCopilotAgent(
        instructions="You answer questions about Azure.",
        default_options=options,
    ) as agent:
        response = await agent.run("What is Azure Service Bus?")
        print(response.text)

asyncio.run(main())
```

---

## 5. `HyperlightExecuteCodeTool` + `AllowedDomain` + `FileMount`

**Source:** `agent_framework_hyperlight._execute_code_tool`, `_types`  
**Package:** `pip install agent-framework[hyperlight]`

`HyperlightExecuteCodeTool` exposes an `execute_code` tool backed by a
**Hyperlight WebAssembly sandbox** — a micro-VM that prevents the LLM-generated Python code
from escaping the container. It extends `FunctionTool` so it slots directly into any agent's
`tools` list.

### Constructor signature

```python
class HyperlightExecuteCodeTool(FunctionTool):
    def __init__(
        self,
        *,
        tools: FunctionTool | Callable | Sequence[...] | None = None,
        approval_mode: ApprovalMode | None = None,        # default "never_require"
        workspace_root: str | Path | None = None,          # mount agent's working dir
        file_mounts: FileMountInput | Sequence[...] | None = None,
        allowed_domains: AllowedDomainInput | Sequence[...] | None = None,
        backend: str = "wasm",
        module: str | None = "python_guest.path",
        module_path: str | None = None,
        _registry: SandboxRuntime | None = None,
    ) -> None: ...
```

### `AllowedDomain` and `FileMount`

```python
class AllowedDomain(NamedTuple):
    target: str                           # e.g. "api.openai.com"
    methods: tuple[str, ...] | None = None  # None = all methods; ("GET",) = read-only

class FileMount(NamedTuple):
    host_path: str | Path   # path on the host
    mount_path: str         # path inside the sandbox (under /input)
```

### Sandbox mutation methods

| Method | Description |
|--------|-------------|
| `add_tools(tools)` | Register sandbox-callable tools inside the Wasm environment |
| `get_tools()` | List currently registered sandbox tools |
| `remove_tool(name)` | Remove one sandbox tool by name |
| `clear_tools()` | Unregister all sandbox tools |
| `add_file_mounts(mounts)` | Mount host paths into `/input/…` inside the sandbox |
| `remove_file_mount(path)` | Remove a mount by its sandbox path |
| `add_allowed_domains(domains)` | Whitelist outbound HTTP targets |
| `remove_allowed_domain(domain)` | Remove one whitelist entry |
| `build_instructions(tools_visible_to_model)` | Get CodeAct instructions for the current config |
| `create_run_tool()` | Snapshot the current config into a new per-run instance |

### Example 1 — sandboxed code execution with tool access

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from agent_framework_hyperlight import HyperlightExecuteCodeTool

@tool
def get_stock_price(ticker: str) -> float:
    """Return the latest stock price for a ticker symbol."""
    prices = {"MSFT": 420.50, "AAPL": 212.30, "GOOGL": 178.00}
    return prices.get(ticker.upper(), 0.0)

execute_tool = HyperlightExecuteCodeTool(
    tools=[get_stock_price],
    allowed_domains=[("finance.yahoo.com", ("GET",))],
)

async def main():
    agent = Agent(
        client=FoundryChatClient(),
        instructions="You are a data-analysis assistant. Use execute_code to compute answers.",
        tools=[execute_tool],
    )
    response = await agent.run(
        "Calculate the average stock price of MSFT, AAPL and GOOGL, "
        "then tell me which is the highest."
    )
    print(response.text)

asyncio.run(main())
```

### Example 2 — file mounts for data ingestion

```python
import asyncio
from pathlib import Path
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_hyperlight import HyperlightExecuteCodeTool, FileMount

# Mount a CSV file from the host into the sandbox
execute_tool = HyperlightExecuteCodeTool(
    file_mounts=[
        FileMount(host_path="/data/sales_2025.csv", mount_path="sales_2025.csv"),
    ],
)

async def main():
    agent = Agent(
        client=FoundryChatClient(),
        instructions="Analyse the mounted sales CSV and produce a summary.",
        tools=[execute_tool],
    )
    response = await agent.run(
        "Load /input/sales_2025.csv using pandas and tell me the top 5 products by revenue."
    )
    print(response.text)

asyncio.run(main())
```

### Example 3 — dynamic tool registration at runtime

```python
from agent_framework import tool
from agent_framework_hyperlight import HyperlightExecuteCodeTool, AllowedDomain

execute_tool = HyperlightExecuteCodeTool()

@tool
def fetch_exchange_rate(currency_pair: str) -> float:
    """Return the current exchange rate for the given pair (e.g. 'USD/EUR')."""
    rates = {"USD/EUR": 0.92, "USD/GBP": 0.79}
    return rates.get(currency_pair, 1.0)

# Add after construction
execute_tool.add_tools(fetch_exchange_rate)
execute_tool.add_allowed_domains([AllowedDomain("api.exchangerate.host", ("GET",))])

print("Tools:", [t.name for t in execute_tool.get_tools()])
print("Domains:", execute_tool.get_allowed_domains())
```

---

## 6. `HyperlightCodeActProvider`

**Source:** `agent_framework_hyperlight._provider`  
**Package:** `pip install agent-framework[hyperlight]`

`HyperlightCodeActProvider` is a `ContextProvider` that owns a `HyperlightExecuteCodeTool`
and injects it — along with CodeAct system-prompt instructions — before every agent run.
The key difference from adding `HyperlightExecuteCodeTool` directly to `tools`:

- The provider creates a **per-run snapshot** (`create_run_tool()`) so each run starts with
  a clean sandbox state.
- It writes CodeAct instructions into `context.extend_instructions()`, keeping the tool's
  description lean and letting the provider manage the full instructions block.

### Constructor signature

```python
class HyperlightCodeActProvider(ContextProvider):
    def __init__(
        self,
        source_id: str = "hyperlight_codeact",
        *,
        tools: FunctionTool | Callable | Sequence[...] | None = None,
        approval_mode: ApprovalMode | None = None,
        workspace_root: str | Path | None = None,
        file_mounts: FileMountInput | Sequence[...] | None = None,
        allowed_domains: AllowedDomainInput | Sequence[...] | None = None,
        backend: str = "wasm",
        module: str | None = "python_guest.path",
        module_path: str | None = None,
        _registry: SandboxRuntime | None = None,
    ) -> None: ...
```

It exposes the same `add_tools`, `get_tools`, `remove_tool`, `clear_tools`,
`add_file_mounts`, `add_allowed_domains` helpers on the provider directly —
they delegate to the internal `HyperlightExecuteCodeTool`.

### Example 1 — provider-owned CodeAct with automatic instructions

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from agent_framework_hyperlight import HyperlightCodeActProvider

@tool
def list_directory(path: str) -> list[str]:
    """List files in a directory."""
    import os
    return os.listdir(path)

provider = HyperlightCodeActProvider(
    tools=[list_directory],
    workspace_root="/workspace",
)

async def main():
    agent = Agent(
        client=FoundryChatClient(),
        instructions="You are a file-system analysis agent.",
        context_providers=[provider],  # injects execute_code tool + CodeAct instructions
    )
    response = await agent.run("Show me the Python files in /workspace and count lines of code.")
    print(response.text)

asyncio.run(main())
```

### Example 2 — inspecting the injected instructions

```python
from agent_framework_hyperlight import HyperlightCodeActProvider

provider = HyperlightCodeActProvider(
    allowed_domains=["api.github.com"],
)

instructions = provider._execute_code_tool.build_instructions(tools_visible_to_model=True)
print(instructions[:500])  # see what CodeAct guidelines are injected
```

---

## 7. `Mem0ContextProvider`

**Source:** `agent_framework_mem0._context_provider`  
**Package:** `pip install agent-framework[mem0]`

`Mem0ContextProvider` integrates [Mem0](https://mem0.ai) — a cloud semantic memory store —
via the `ContextProvider` hooks pattern. Before each run it **searches** Mem0 for memories
relevant to the user's input and injects them as context; after each run it **stores** the
conversation for future retrieval.

### Constructor signature

```python
class Mem0ContextProvider(ContextProvider):
    DEFAULT_CONTEXT_PROMPT = "## Memories\nConsider the following memories when answering user questions:"
    DEFAULT_SOURCE_ID = "mem0"

    def __init__(
        self,
        source_id: str = DEFAULT_SOURCE_ID,
        mem0_client: AsyncMemory | AsyncMemoryClient | None = None,
        api_key: str | None = None,
        application_id: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        *,
        context_prompt: str | None = None,
    ) -> None: ...
```

At least one of `agent_id`, `user_id`, or `application_id` must be set — they scope which
memories are searched and where new memories are stored.

### Lifecycle hooks

| Hook | Triggered | What it does |
|------|-----------|--------------|
| `before_run` | Before the LLM call | Searches Mem0 with the user's input text; injects matching memories via `context.extend_messages` |
| `after_run` | After the LLM responds | Stores the full user + assistant turn via `mem0_client.add` |

### Example 1 — per-user persistent memory with Mem0 Cloud

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_mem0 import Mem0ContextProvider

async def run_for_user(user_id: str, prompt: str):
    provider = Mem0ContextProvider(
        api_key="mem0_api_key_here",
        user_id=user_id,
    )
    async with provider:
        agent = Agent(
            client=FoundryChatClient(),
            instructions="You are a personal assistant who remembers user preferences.",
            context_providers=[provider],
        )
        response = await agent.run(prompt)
        print(response.text)

async def main():
    await run_for_user("alice", "I prefer concise answers in bullet points.")
    await run_for_user("alice", "What format do I prefer for answers?")

asyncio.run(main())
```

### Example 2 — scoped by agent_id for team-shared knowledge

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_mem0 import Mem0ContextProvider

async def main():
    provider = Mem0ContextProvider(
        api_key="mem0_api_key_here",
        agent_id="knowledge-base-agent",   # shared across all users of this agent
        context_prompt="## Shared Knowledge\nUse the following team knowledge:",
    )
    async with provider:
        agent = Agent(
            client=FoundryChatClient(),
            instructions="You are a team knowledge assistant.",
            context_providers=[provider],
        )
        # First user stores knowledge
        await agent.run("The deployment pipeline uses GitHub Actions with self-hosted runners.")
        # Later user retrieves it
        response = await agent.run("How does our deployment pipeline work?")
        print(response.text)

asyncio.run(main())
```

### Example 3 — bring your own OSS Mem0 client

```python
import asyncio
from mem0 import AsyncMemory
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_mem0 import Mem0ContextProvider

async def main():
    # OSS Mem0 — stores locally (no cloud API key needed)
    mem0_client = AsyncMemory()

    provider = Mem0ContextProvider(
        mem0_client=mem0_client,
        user_id="local_user",
    )
    async with provider:
        agent = Agent(
            client=FoundryChatClient(),
            instructions="You are a helpful assistant.",
            context_providers=[provider],
        )
        response = await agent.run("My name is Alex and I'm a Python developer.")
        print(response.text)

asyncio.run(main())
```

---

## 8. `RedisContextProvider` + `RedisHistoryProvider`

**Source:** `agent_framework_redis._context_provider`, `_history_provider`  
**Package:** `pip install agent-framework[redis]`

These two providers replace the in-process `MemoryContextProvider` / `FileHistoryProvider`
with a **Redis backend** suited for multi-instance or serverless deployments.

| Class | Extends | What it stores |
|-------|---------|---------------|
| `RedisContextProvider` | `ContextProvider` | Semantic context (supports optional vector search via `redisvl`) |
| `RedisHistoryProvider` | `HistoryProvider` | Serialised `Message` objects in Redis Lists, keyed by session ID |

### `RedisContextProvider` constructor (key params)

```python
class RedisContextProvider(ContextProvider):
    def __init__(
        self,
        source_id: str = "redis",
        redis_url: str = "redis://localhost:6379",
        index_name: str = "context",
        prefix: str = "context",
        *,
        redis_vectorizer: BaseVectorizer | None = None,   # enables hybrid vector search
        vector_field_name: str | None = None,
        vector_algorithm: Literal["flat", "hnsw"] | None = None,
        vector_distance_metric: Literal["cosine", "ip", "l2"] | None = None,
        application_id: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        context_prompt: str | None = None,
        overwrite_index: bool = False,
    ) -> None: ...
```

### `RedisHistoryProvider` constructor (key params)

```python
class RedisHistoryProvider(HistoryProvider):
    def __init__(
        self,
        source_id: str = "redis_memory",
        redis_url: str | None = None,          # e.g. "redis://localhost:6379"
        credential_provider: CredentialProvider | None = None,  # Azure AD auth
        host: str | None = None,               # required with credential_provider
        port: int = 6380,                      # Azure Redis SSL default
        ssl: bool = True,
        username: str | None = None,
        *,
        key_prefix: str = "chat_messages",
        max_messages: int | None = None,       # None = unlimited; set to bound memory
        load_messages: bool = True,
        store_outputs: bool = True,
        store_inputs: bool = True,
        store_context_messages: bool = False,
    ) -> None: ...
```

### Example 1 — Redis-backed conversation history for multi-instance deployment

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_redis import RedisHistoryProvider

async def handle_request(session_id: str, user_message: str) -> str:
    # Each request may run on a different pod — Redis persists the history
    history_provider = RedisHistoryProvider(
        redis_url="redis://redis-service:6379",
        key_prefix="chat",
        max_messages=50,    # keep the last 50 messages per session
    )

    agent = Agent(
        client=FoundryChatClient(),
        instructions="You are a customer support agent.",
        context_providers=[history_provider],
    )

    session = agent.create_session(session_id=session_id)
    response = await agent.run(user_message, session=session)
    return response.text

async def main():
    # Simulate two requests in the same logical session on different pods
    session_id = "customer-12345"
    r1 = await handle_request(session_id, "I can't log in to my account.")
    print(f"Pod A: {r1}")

    r2 = await handle_request(session_id, "What was my original issue?")
    print(f"Pod B: {r2}")  # will recall the login issue from Redis

asyncio.run(main())
```

### Example 2 — Azure Cache for Redis with managed identity

```python
import asyncio
from redis.asyncio.connection import SSLConnection
from redis.asyncio.client import ConnectionPool
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_redis import RedisHistoryProvider

# azure-redis-py credential provider wraps DefaultAzureCredential
from azure.identity import DefaultAzureCredential
from redis.asyncio.credentials import CredentialProvider

class AzureRedisCredentialProvider(CredentialProvider):
    def __init__(self, credential):
        self._credential = credential

    def get_credentials(self) -> tuple[str, str]:
        token = self._credential.get_token("https://redis.azure.com/.default")
        return "", token.token  # username is empty for Azure Redis MI

async def main():
    cred_provider = AzureRedisCredentialProvider(DefaultAzureCredential())
    history = RedisHistoryProvider(
        credential_provider=cred_provider,
        host="my-cache.redis.cache.windows.net",
        port=6380,
        ssl=True,
        max_messages=100,
    )

    agent = Agent(
        client=FoundryChatClient(),
        instructions="You are an enterprise assistant.",
        context_providers=[history],
    )
    response = await agent.run("What is our data retention policy?")
    print(response.text)

asyncio.run(main())
```

### Example 3 — semantic context retrieval with vector search

```python
import asyncio
from redisvl.utils.vectorize import HFTextVectorizer
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_redis import RedisContextProvider

async def main():
    vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")

    context_provider = RedisContextProvider(
        redis_url="redis://localhost:6379",
        index_name="company_knowledge",
        prefix="kb",
        redis_vectorizer=vectorizer,
        vector_field_name="embedding",
        vector_algorithm="hnsw",
        vector_distance_metric="cosine",
        agent_id="knowledge-agent",
    )

    agent = Agent(
        client=FoundryChatClient(),
        instructions="Answer questions using the retrieved context.",
        context_providers=[context_provider],
    )
    response = await agent.run("What is our SLA for critical incidents?")
    print(response.text)

asyncio.run(main())
```

---

## 9. `StandardMagenticManager` + `MagenticContext` + `MagenticProgressLedger`

**Source:** `agent_framework_orchestrations._magentic`  
**Package:** `pip install agent-framework[orchestrations]`

`StandardMagenticManager` implements the **Magentic-One** multi-agent orchestration algorithm.
An orchestrator agent manages a *task ledger* (facts + plan) and a *progress ledger* (JSON
tracking who acts next and whether the task is stalling), delegating work rounds to specialist
sub-agents.

### Class hierarchy

```
MagenticManagerBase (abstract)
└── StandardMagenticManager   # real LLM calls via Agent
```

### `StandardMagenticManager` constructor

```python
class StandardMagenticManager(MagenticManagerBase):
    def __init__(
        self,
        agent: SupportsAgentRun,           # orchestrator agent for LLM calls
        task_ledger: _MagenticTaskLedger | None = None,
        *,
        task_ledger_facts_prompt: str | None = None,
        task_ledger_plan_prompt: str | None = None,
        task_ledger_full_prompt: str | None = None,
        task_ledger_facts_update_prompt: str | None = None,
        task_ledger_plan_update_prompt: str | None = None,
        progress_ledger_prompt: str | None = None,
        final_answer_prompt: str | None = None,
        max_stall_count: int = 3,
        max_reset_count: int | None = None,
        max_round_count: int | None = None,
        progress_ledger_retry_count: int | None = None,  # default 3
    ) -> None: ...
```

### `MagenticContext` fields

```python
@dataclass
class MagenticContext:
    task: str
    chat_history: list[Message]                   # grows with every round
    participant_descriptions: dict[str, str]       # agent_name → description
    round_count: int
    stall_count: int
    reset_count: int

    def reset(self) -> None: ...    # clears history, resets stall_count, increments reset_count
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> MagenticContext: ...
```

### `MagenticProgressLedger` fields

```python
@dataclass
class MagenticProgressLedger:
    is_request_satisfied: MagenticProgressLedgerItem   # .answer is bool
    is_in_loop: MagenticProgressLedgerItem             # .answer is bool
    is_progress_being_made: MagenticProgressLedgerItem # .answer is bool
    next_speaker: MagenticProgressLedgerItem           # .answer is str (agent name)
    instruction_or_question: MagenticProgressLedgerItem  # .answer is str (prompt for next agent)

@dataclass
class MagenticProgressLedgerItem:
    reason: str
    answer: str | bool
```

### Example 1 — three-agent Magentic-One workflow

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_orchestrations import MagenticBuilder

async def main():
    client = FoundryChatClient()

    # Orchestrator uses StandardMagenticManager under the hood
    orchestrator = Agent(client=client, name="orchestrator",
                         instructions="You are the Magentic orchestrator.")
    researcher   = Agent(client=client, name="researcher",
                         instructions="You research and gather facts.")
    coder        = Agent(client=client, name="coder",
                         instructions="You write Python code to solve problems.")
    critic       = Agent(client=client, name="critic",
                         instructions="You review and improve solutions.")

    runner = (
        MagenticBuilder()
        .add_participant(researcher)
        .add_participant(coder)
        .add_participant(critic)
        .with_manager(orchestrator)
        .build()
    )

    response = await runner.run(
        "Write a Python script that computes the first 20 Fibonacci numbers "
        "and saves them to a CSV file."
    )
    print(response.text)

asyncio.run(main())
```

### Example 2 — custom manager with prompt overrides

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_orchestrations import MagenticBuilder
from agent_framework_orchestrations._magentic import StandardMagenticManager

TERSE_PROGRESS_PROMPT = """
Review the conversation for task: {task}
Team: {team}
Eligible speakers: {names}
Reply ONLY with a valid JSON progress ledger.
"""

async def main():
    client = FoundryChatClient()
    orchestrator = Agent(client=client, name="mgr",
                         instructions="You orchestrate the team concisely.")
    analyst = Agent(client=client, name="analyst",
                    instructions="You analyse data and produce insights.")

    manager = StandardMagenticManager(
        agent=orchestrator,
        max_stall_count=2,
        max_round_count=10,
        progress_ledger_prompt=TERSE_PROGRESS_PROMPT,
    )

    runner = (
        MagenticBuilder()
        .add_participant(analyst)
        .with_custom_manager(manager)
        .build()
    )

    response = await runner.run("Analyse the top 3 reasons for customer churn in SaaS.")
    print(response.text)

asyncio.run(main())
```

### Example 3 — inspecting `MagenticContext` mid-run

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_orchestrations._magentic import (
    StandardMagenticManager,
    MagenticContext,
)

async def main():
    client = FoundryChatClient()
    orchestrator = Agent(client=client, name="orchestrator",
                         instructions="You orchestrate the conversation.")
    manager = StandardMagenticManager(agent=orchestrator, max_round_count=5)

    # Manually construct a context to inspect serialisation
    ctx = MagenticContext(
        task="Build a REST API for a todo list",
        participant_descriptions={
            "designer": "Designs the API endpoints and contracts.",
            "developer": "Implements the API in Python.",
        },
    )
    print("Initial context:", ctx.to_dict())
    ctx.stall_count += 1
    ctx.reset()
    print("After reset:", ctx.to_dict())  # stall_count=0, reset_count=1

asyncio.run(main())
```

---

## 10. `FileSkillsSource` + `FileSkill` + `FilteringSkillsSource` + `AggregatingSkillsSource`

**Source:** `agent_framework._skills`  
**Package:** `agent-framework` (core — no extras needed)

The skills subsystem lets agents discover and use reusable **prompt bundles** stored as
`SKILL.md` files on disk. `FileSkillsSource` discovers them, `FilteringSkillsSource` gates
by capability, and `AggregatingSkillsSource` fans out across multiple sources.

### `FileSkillsSource` constructor

```python
class FileSkillsSource(SkillsSource):
    def __init__(
        self,
        skill_paths: str | Path | Sequence[str | Path],
        *,
        script_runner: SkillScriptRunner | None = None,
        resource_extensions: tuple[str, ...] | None = None,
        # default: (".md", ".json", ".yaml", ".yml", ".csv", ".xml", ".txt")
        script_extensions: tuple[str, ...] | None = None,
        # default: (".py",)
        resource_directories: Sequence[str] | None = None,
        # default: ("references", "assets")  — per agentskills.io spec
        script_directories: Sequence[str] | None = None,
        # default: ("scripts",)
    ) -> None: ...
```

### `SKILL.md` file format

A minimal skill directory on disk looks like:

```
skills/
  summarise/
    SKILL.md           ← frontmatter + prompt content
    references/
      example.md       ← auto-discovered resource
    scripts/
      run.py           ← auto-discovered script
```

```markdown
---
name: summarise
description: Summarise long documents concisely
tags: ["nlp", "summarisation"]
---

You are a summarisation expert. Given the following document,
produce a 3-bullet executive summary:

{{document}}
```

### Composition helpers

| Class | Description |
|-------|-------------|
| `FilteringSkillsSource` | Wraps another `SkillsSource` and filters by a predicate (tags, name patterns, etc.) |
| `AggregatingSkillsSource` | Merges skills from multiple `SkillsSource` instances, deduplicating by name |
| `DeduplicatingSkillsSource` | Like `AggregatingSkillsSource` but with explicit deduplication strategy |
| `InMemorySkillsSource` | In-process source for programmatically constructed skills (no files needed) |

### Example 1 — load skills from a directory

```python
import asyncio
from agent_framework import Agent
from agent_framework._skills import FileSkillsSource
from agent_framework.foundry import FoundryChatClient

async def main():
    source = FileSkillsSource(skill_paths="./skills")
    skills = await source.get_skills()
    print(f"Loaded {len(skills)} skills:")
    for skill in skills:
        print(f"  {skill.frontmatter.name}: {skill.frontmatter.description}")

    agent = Agent(
        client=FoundryChatClient(),
        instructions="Use the available skills to help users.",
    )
    # Pass skills to your skill-aware context provider or agent configuration
    for skill in skills:
        print(skill.content)  # the prompt template

asyncio.run(main())
```

### Example 2 — multiple skill directories merged with `AggregatingSkillsSource`

```python
import asyncio
from agent_framework._skills import FileSkillsSource, AggregatingSkillsSource

async def main():
    core_source   = FileSkillsSource(skill_paths="./skills/core")
    domain_source = FileSkillsSource(skill_paths=["./skills/finance", "./skills/legal"])

    combined = AggregatingSkillsSource([core_source, domain_source])
    skills = await combined.get_skills()

    print(f"Total skills: {len(skills)}")

asyncio.run(main())
```

### Example 3 — tag-filtered skills for a specialist agent

```python
import asyncio
from agent_framework._skills import FileSkillsSource, FilteringSkillsSource

async def main():
    all_skills = FileSkillsSource(skill_paths="./skills")

    # Only surface skills tagged "coding"
    coding_skills = FilteringSkillsSource(
        source=all_skills,
        filter_func=lambda skill: "coding" in (skill.frontmatter.tags or []),
    )

    skills = await coding_skills.get_skills()
    print("Coding skills:", [s.frontmatter.name for s in skills])

asyncio.run(main())
```

### Example 4 — custom resource and script directories

```python
import asyncio
from agent_framework._skills import FileSkillsSource

async def main():
    # Non-standard layout: docs/ for resources, actions/ for scripts,
    # plus root-level files (".")
    source = FileSkillsSource(
        skill_paths="./company-skills",
        resource_directories=[".", "docs", "templates"],
        script_directories=["actions"],
        resource_extensions=(".md", ".txt", ".json"),
        script_extensions=(".py", ".sh"),
    )
    skills = await source.get_skills()
    for skill in skills:
        print(f"{skill.frontmatter.name}: "
              f"{len(skill.resources)} resources, "
              f"{len(skill.scripts)} scripts")

asyncio.run(main())
```

---

## Upgrade notes — integration packages at 1.8.0

| Package | Version | New in 1.8.0 |
|---------|---------|--------------|
| `agent-framework[ollama]` | 1.8.0 | `think` option for reasoning models; `keep_alive` control; native `response_format` schema |
| `agent-framework[purview]` | 1.8.0 | `PurviewChatPolicyMiddleware` (chat-level variant); `PurviewAppLocation`; `ignore_payment_required` flag |
| `agent-framework[durabletask]` | 1.8.0 | `DurableAIAgentWorker.registered_agent_names` property; per-agent callback override |
| `agent-framework[github-copilot]` | 1.8.0 | `on_function_approval` callback; `instruction_directories`; BYOK `provider` config |
| `agent-framework[hyperlight]` | 1.8.0 | `HyperlightCodeActProvider` for provider-owned CodeAct; `AllowedDomain.methods` restriction |
| `agent-framework[mem0]` | 1.8.0 | `AsyncMemory` OSS client support; `application_id` scoping |
| `agent-framework[redis]` | 1.8.0 | `RedisHistoryProvider` Azure AD `credential_provider` path; `store_context_from` filter; `vector_algorithm` / `vector_distance_metric` options |
| `agent-framework[orchestrations]` | 1.8.0 | `MagenticProgressLedger` retry logic; `StandardMagenticManager.progress_ledger_retry_count`; `MagenticContext.from_dict` / `to_dict` |
| `agent-framework` (core) | 1.8.0 | `FileSkillsSource` path-traversal guard; `FilteringSkillsSource`; `AggregatingSkillsSource` |

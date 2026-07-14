---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 38"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.11.0: AnthropicVertexClient+RawAnthropicVertexClient (Claude on Google Vertex AI — region/project_id/credentials wiring, access_token refresh, middleware-free raw variant); GAIA+Task+Prediction+Evaluation+GAIATelemetryConfig (GAIA benchmark harness — custom TaskRunner protocol, Evaluator protocol, OTel trace-to-file, HuggingFace dataset download, concurrent task execution); AgentFrameworkException hierarchy (18-class exception taxonomy — inner_exception chaining, log_level suppression, UserInputRequiredException.contents propagation, WorkflowCheckpointException, MiddlewareException, IntegrationException tree); ExperimentalFeature+ReleaseCandidateFeature+FeatureStageWarning (feature-stage enums — 12 EXPERIMENTAL IDs, RC empty inventory in 1.11.0, warnings.filterwarnings suppression, getattr guard pattern); AgentSession (lightweight session state — session_id UUID auto-gen, service_session_id ServiceSessionId, state dict SerializationProtocol round-trip, to_dict/from_dict persistence); AgentResponse+AgentResponseUpdate (agent run result model — messages list, usage_details, finish_reason, value structured output, from_updates streaming assembly, to_json/from_json serialisation); ChatContext (chat middleware pipeline context — messages/options/metadata mutation, result override, stream_transform_hooks PII redaction, stream_result_hooks, function_invocation_kwargs); ResponseStream (async stream wrapper — finalizer, transform_hooks per-update, result_hooks post-finalize, cleanup_hooks, get_final_response() on-demand); BaseEmbeddingClient+GeneratedEmbeddings+EmbeddingGenerationOptions (custom embedding backend — abstract get_embeddings, Embedding vector wrapper, usage dict, OTEL_PROVIDER_NAME hook); BaseChatClient (custom LLM provider — _inner_get_response streaming/non-streaming, get_response typed overloads, to_dict/from_dict for config serialisation) — source-verified at agent-framework 1.11.0."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 61
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 38

Verified against **agent-framework 1.11.0** (installed July 2026). Every constructor signature, parameter description, and code example was derived from the installed package source using `inspect.getsource()`.

Sub-packages introspected:
`agent_framework.google`,
`agent_framework.lab.gaia`,
`agent_framework.exceptions`,
`agent_framework._feature_stage`,
`agent_framework._sessions`,
`agent_framework._clients`.

**Previous volumes:** [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) through [Vol. 37](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v37/) — 370+ classes covered.

This volume covers **ten class groups** across the Google Vertex AI integration, the GAIA benchmark harness, the complete exception hierarchy, feature-staging enums, session and response models, the chat middleware context, streaming primitives, and custom provider base classes.

| # | Class / group | Module |
|---|---|---|
| 1 | `AnthropicVertexClient` · `RawAnthropicVertexClient` | `agent_framework.google` |
| 2 | `GAIA` · `Task` · `Prediction` · `Evaluation` · `GAIATelemetryConfig` | `agent_framework.lab.gaia` |
| 3 | `AgentFrameworkException` hierarchy (18 classes) | `agent_framework.exceptions` |
| 4 | `ExperimentalFeature` · `ReleaseCandidateFeature` · `FeatureStageWarning` | `agent_framework._feature_stage` |
| 5 | `AgentSession` | `agent_framework._sessions` |
| 6 | `AgentResponse` · `AgentResponseUpdate` | `agent_framework._sessions` |
| 7 | `ChatContext` | `agent_framework._sessions` |
| 8 | `ResponseStream` | `agent_framework._clients` |
| 9 | `BaseEmbeddingClient` · `GeneratedEmbeddings` · `EmbeddingGenerationOptions` | `agent_framework._clients` |
| 10 | `BaseChatClient` | `agent_framework._clients` |

---

## 1 · Google Vertex AI — Anthropic Client

**Module:** `agent_framework.google`
**Install:** `pip install agent-framework agent-framework-anthropic "anthropic[vertex]"`
*(provider package `agent-framework-anthropic` wires the Anthropic SDK into the framework; `anthropic[vertex]` adds `google-auth` for Vertex AI authentication)*

The `google` sub-package provides two classes for running Anthropic (Claude) models through **Google Vertex AI** — a fully-featured client with the standard middleware stack, and a raw variant without it.

> **Import note:** Both classes are also importable from `agent_framework.anthropic` (e.g. `from agent_framework.anthropic import AnthropicVertexClient`) — the two paths resolve to the same class. Other volumes in this guide use `agent_framework.anthropic`; `agent_framework.google` is the namespace grouping used in this section.

### `RawAnthropicVertexClient`

Thin wrapper around `anthropic.AsyncAnthropicVertex` that maps the framework's `Message`/`ChatOptions` contract onto the Vertex REST API.  Sets `OTEL_PROVIDER_NAME = "google.vertex.ai"` so telemetry labels route to the correct provider bucket.

```python
Constructor:
RawAnthropicVertexClient(
    *,
    model: str | None = None,          # e.g. "claude-opus-4-8@20251101"
    region: str | None = None,          # falls back to CLOUD_ML_REGION env var
    project_id: str | None = None,      # falls back to ANTHROPIC_VERTEX_PROJECT_ID
    access_token: str | None = None,    # explicit OAuth2 access token
    credentials: GoogleCredentials | None = None,  # google.oauth2 credentials object
    base_url: str | None = None,
    anthropic_client: AsyncAnthropicVertex | None = None,  # reuse existing client
    additional_beta_flags: list[str] | None = None,
    additional_properties: dict[str, Any] | None = None,
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
)
```

### `AnthropicVertexClient`

Composes the MRO stack `FunctionInvocationLayer → ChatMiddlewareLayer → ChatTelemetryLayer → RawAnthropicVertexClient` and accepts `middleware=` and `function_invocation_configuration=` to allow the same middleware and tool patterns used with every other provider.

```python
Constructor:
AnthropicVertexClient(
    # All fields from RawAnthropicVertexClient, plus:
    middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
    function_invocation_configuration: FunctionInvocationConfiguration | None = None,
)
```

**Example 1 — basic usage with env vars**

```python
import asyncio
import os
from agent_framework import Agent, tool
from agent_framework.google import AnthropicVertexClient

# Set env vars before running:
#   CLOUD_ML_REGION=us-east5
#   ANTHROPIC_VERTEX_PROJECT_ID=my-gcp-project
#   ANTHROPIC_API_KEY=<not used but still read by anthropic SDK>

@tool
def get_city_temperature(city: str) -> str:
    """Return the current temperature for a city."""
    temperatures = {"London": "12°C", "New York": "24°C", "Tokyo": "29°C"}
    return temperatures.get(city, "unknown")

async def main():
    client = AnthropicVertexClient(
        model="claude-opus-4-8@20251101",
        region=os.environ.get("CLOUD_ML_REGION", "us-east5"),
        project_id=os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"),
    )
    agent = Agent(
        client=client,
        name="weather-agent",
        instructions="You help users check weather.",
        tools=[get_city_temperature],
    )
    response = await agent.run("What is the temperature in London?")
    print(response.text)

# asyncio.run(main())
```

**Example 2 — explicit `google.oauth2` credentials**

```python
import asyncio
from google.oauth2 import service_account
from agent_framework import Agent
from agent_framework.google import AnthropicVertexClient

async def main():
    creds = service_account.Credentials.from_service_account_file(
        "service-account.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = AnthropicVertexClient(
        model="claude-haiku-4-5-20251001",
        region="europe-west4",
        project_id="my-gcp-project",
        credentials=creds,
    )
    agent = Agent(client=client, name="assistant", instructions="Be concise.")
    response = await agent.run("Summarise the Vertex AI platform in two sentences.")
    print(response.text)

# asyncio.run(main())
```

**Example 3 — raw client for latency-critical paths**

```python
import asyncio
from agent_framework import Message
from agent_framework.google import RawAnthropicVertexClient

async def main():
    # RawAnthropicVertexClient skips middleware + telemetry — useful when
    # you manage your own OTel spans or need the absolute minimum round-trip.
    raw_client = RawAnthropicVertexClient(
        model="claude-haiku-4-5-20251001",
        region="us-east5",
        project_id="my-gcp-project",
    )
    messages = [Message(role="user", contents=["Hello from Vertex!"])]
    response = await raw_client.get_response(messages=messages)
    print(response.messages[0].content_str)

# asyncio.run(main())
```

---

## 2 · GAIA Benchmark Harness

**Module:** `agent_framework.lab.gaia`
**Install:** `pip install "agent-framework-lab[gaia]"`
*(adds `huggingface-hub`, `opentelemetry-sdk`, `pydantic`, `tqdm`, `orjson`, and `pyarrow` needed by the GAIA runner)*

The `gaia` lab module provides types and a runner for the **GAIA** (General AI Assistant) benchmark — a curated set of real-world tasks requiring multi-step reasoning with tool use.

### Key types

```
Task          @dataclass — task_id, question, answer, level (1-3), file_name, metadata
Prediction    @dataclass — prediction: str, messages: list[Any], metadata: dict
Evaluation    @dataclass — is_correct: bool, score: float, details: dict
TaskRunner    Protocol  — async def __call__(task: Task) -> Prediction
Evaluator     Protocol  — async def __call__(task, prediction) -> Evaluation
```

### `GAIA`

```python
Constructor:
GAIA(
    evaluator: Evaluator | None = None,      # None → default exact-match scorer
    data_dir: str | None = None,             # cache location; default is a temp dir
    hf_token: str | None = None,             # HuggingFace token (or HF_TOKEN env var)
    telemetry_config: GAIATelemetryConfig | None = None,
)

Key methods:
  await gaia.run(task_runner, level=1, max_n=None,
                 parallel=1, timeout=None, out=None) -> list[TaskResult]
  await gaia.run_task(task_runner, task, timeout=300) -> TaskResult
```

### `GAIATelemetryConfig`

```python
Constructor:
GAIATelemetryConfig(
    enable_tracing: bool = False,
    otlp_endpoint: str | None = None,  # OTLP gRPC/HTTP endpoint
    trace_to_file: bool = False,
    file_path: str | None = None,      # default: gaia_traces.json
)
```

**Example 4 — custom `TaskRunner` using an agent**

```python
import asyncio, os
from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from agent_framework.lab.gaia import GAIA, Task, Prediction, TaskRunner

@tool
def web_search(query: str) -> str:
    """Stub web search — replace with real implementation."""
    return f"[search results for '{query}']"

class AgentTaskRunner:
    """Adapts an agent to the TaskRunner protocol."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    async def __call__(self, task: Task) -> Prediction:
        prompt = task.question
        if task.file_name:
            prompt += f"\n\n[Attachment: {task.file_name}]"
        response = await self._agent.run(prompt)
        return Prediction(
            prediction=response.text,
            messages=[m.to_dict() for m in response.messages],
        )

async def main():
    client = FoundryChatClient(model="gpt-4o")
    agent = Agent(
        client=client,
        name="gaia-solver",
        instructions="You are a general AI assistant. Use the web_search tool to find answers.",
        tools=[web_search],
    )
    runner = AgentTaskRunner(agent)

    gaia = GAIA(hf_token=os.environ["HF_TOKEN"])

    # Run only the first 5 level-1 tasks
    results = await gaia.run(
        runner,
        level=1,
        max_n=5,
        parallel=2,
        timeout=120,
    )
    correct = sum(1 for r in results if r.evaluation and r.evaluation.is_correct)
    print(f"Score: {correct}/{len(results)}")

# asyncio.run(main())
```

**Example 5 — custom `Evaluator` for fuzzy matching**

```python
import asyncio
from agent_framework.lab.gaia import GAIA, Task, Prediction, Evaluation

async def fuzzy_evaluator(task: Task, prediction: Prediction) -> Evaluation:
    """Accept the prediction if the correct answer appears anywhere in it."""
    if task.answer is None:
        return Evaluation(is_correct=False, score=0.0)
    expected = task.answer.strip().lower()
    predicted = prediction.prediction.strip().lower()
    is_correct = expected in predicted
    return Evaluation(
        is_correct=is_correct,
        score=1.0 if is_correct else 0.0,
        details={"expected": expected, "predicted_excerpt": predicted[:100]},
    )

async def main():
    gaia = GAIA(evaluator=fuzzy_evaluator)
    # Run a single synthetic task (no HF download needed)
    task = Task(task_id="test-1", question="What is 2 + 2?", answer="4")
    prediction = Prediction(prediction="The answer is 4.", messages=[])
    evaluation = await gaia.evaluator(task, prediction)
    print(evaluation)  # Evaluation(is_correct=True, score=1.0, ...)

asyncio.run(main())
```

**Example 6 — OTel file tracing**

```python
import asyncio
from agent_framework.lab.gaia import GAIA, GAIATelemetryConfig

async def main():
    telemetry = GAIATelemetryConfig(
        enable_tracing=True,
        trace_to_file=True,
        file_path="gaia_traces.json",
    )
    gaia = GAIA(telemetry_config=telemetry)
    print("Tracing enabled:", telemetry.enable_tracing)
    print("Trace file:", telemetry.file_path)
    # Traces are written to gaia_traces.json after each task run

asyncio.run(main())
```

---

## 3 · Exception Hierarchy

**Module:** `agent_framework.exceptions`
**Install:** `pip install agent-framework`

The exception tree has **18 public classes** across two root branches. Understanding the hierarchy lets you write narrow `except` clauses and distinguish provider-level errors from agent-level errors.

```
Exception
└── AgentFrameworkException         # base: inner_exception chaining + auto-log
    ├── AgentException              # agent-level errors
    │   ├── AgentContentFilterException
    │   ├── AgentInvalidAuthException
    │   ├── AgentInvalidRequestException
    │   └── AgentInvalidResponseException
    ├── ChatClientException         # provider-level errors (chat clients)
    │   ├── ChatClientContentFilterException
    │   ├── ChatClientInvalidAuthException
    │   ├── ChatClientInvalidRequestException
    │   └── ChatClientInvalidResponseException
    ├── IntegrationException        # external service errors
    │   ├── IntegrationContentFilterException
    │   ├── IntegrationInitializationError
    │   ├── IntegrationInvalidAuthException
    │   ├── IntegrationInvalidRequestException
    │   └── IntegrationInvalidResponseException
    ├── MiddlewareException         # middleware pipeline errors
    ├── SettingNotFoundError        # missing required setting
    ├── ToolException               # tool-level errors
    │   ├── ToolExecutionException  # runtime tool failure
    │   └── UserInputRequiredException  # sub-agent needs human input
    ├── WorkflowException
    │   └── WorkflowRunnerException
    │       ├── WorkflowCheckpointException
    │       └── WorkflowConvergenceException
    └── ContentError
```

### `AgentFrameworkException`

```python
Constructor:
AgentFrameworkException(
    message: str,
    inner_exception: Exception | None = None,
    log_level: int | None = 10,  # logging.DEBUG; pass None to suppress logging
    *args, **kwargs,
)
```

### `UserInputRequiredException`

```python
Constructor:
UserInputRequiredException(
    contents: list[Any],  # user-input-request Content items from a sub-agent
    message: str = "Tool requires user input to proceed.",
)
# Access via: exc.contents
```

**Example 7 — narrow exception handling by layer**

```python
import asyncio
from agent_framework import Agent
from agent_framework.exceptions import (
    AgentContentFilterException,
    ChatClientInvalidAuthException,
    ChatClientContentFilterException,
    WorkflowException,
    MiddlewareException,
)

async def safe_run(agent: Agent, prompt: str) -> str:
    try:
        response = await agent.run(prompt)
        return response.text
    except AgentContentFilterException:
        return "[Blocked by agent content filter]"
    except ChatClientContentFilterException:
        return "[Blocked by provider content filter]"
    except ChatClientInvalidAuthException:
        return "[Authentication failed — check API key]"
    except MiddlewareException as exc:
        return f"[Middleware error: {exc}]"
    except WorkflowException as exc:
        return f"[Workflow error: {exc}]"

# asyncio.run(safe_run(your_agent, "..."))
```

**Example 8 — inner exception chaining and log suppression**

```python
from agent_framework.exceptions import AgentFrameworkException, IntegrationException

try:
    # Simulate an integration failure
    raise ValueError("Connection refused")
except ValueError as original:
    # inner_exception is stored and included in __cause__
    raise IntegrationException(
        "Failed to connect to vector store",
        inner_exception=original,
        log_level=None,  # suppress automatic debug log
    ) from original
```

**Example 9 — `UserInputRequiredException` propagation**

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.exceptions import UserInputRequiredException

@tool
async def require_oauth(scope: str) -> str:
    """A tool that delegates OAuth consent to the end user."""
    # In a real scenario, the sub-agent raises this with oauth_consent_request content
    raise UserInputRequiredException(
        contents=[{"type": "oauth_consent_request", "scope": scope}],
        message=f"OAuth consent required for scope: {scope}",
    )

async def handle_run(agent: Agent) -> None:
    try:
        await agent.run("Connect to Google Calendar with read scope.")
    except UserInputRequiredException as exc:
        for content in exc.contents:
            if content.get("type") == "oauth_consent_request":
                print(f"Please grant access: scope={content.get('scope')}")

# asyncio.run(handle_run(your_agent))
```

---

## 4 · Feature Staging

**Module:** `agent_framework._feature_stage`
**Install:** `pip install agent-framework`

The feature-staging module tracks which APIs are **experimental** or **release candidate**. This lets you opt into experimental features deliberately and suppress their warnings in tests.

### `ExperimentalFeature` (str, Enum)

Current members in 1.11.0:
```
DECLARATIVE_AGENTS     EVALS              FILE_HISTORY
FIDES                  FOUNDRY_TOOLS      FOUNDRY_PREVIEW_TOOLS
FUNCTIONAL_WORKFLOWS   HARNESS            MCP_LONG_RUNNING_TASKS
MCP_SKILLS             PROGRESSIVE_TOOLS  TO_PROMPT_AGENT
```

### `ReleaseCandidateFeature` (str, Enum)

Empty in 1.11.0 — all prior RCs have graduated to stable.

### `FeatureStageWarning` / `ExperimentalWarning`

```
FutureWarning
└── FeatureStageWarning   # base warning category
    └── ExperimentalWarning  # emitted when an experimental API is called
```

**Example 10 — suppress experimental warnings in tests**

```python
import warnings
from agent_framework._feature_stage import ExperimentalWarning

# In a pytest conftest.py or test setup:
def pytest_configure(config):
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
```

**Example 11 — check membership safely**

```python
from agent_framework._feature_stage import ExperimentalFeature

# The enum inventory changes across versions — use getattr guard
feature_id = "HARNESS"
is_experimental = feature_id in {f.value for f in ExperimentalFeature}
print(f"HARNESS is experimental: {is_experimental}")  # True in 1.11.0

# DO NOT rely on enum membership being stable across versions
# getattr guard for optional __feature_id__ metadata
for feature in ExperimentalFeature:
    fid = getattr(feature, "__feature_id__", None)
    print(f"{feature.value}: __feature_id__={fid}")
```

**Example 12 — catch warnings selectively in production code**

```python
import warnings
from agent_framework._feature_stage import ExperimentalWarning, FeatureStageWarning

# Promote experimental warnings to errors in CI
def enable_strict_feature_warnings():
    warnings.filterwarnings("error", category=ExperimentalWarning)

# Re-allow for a specific block
def use_experimental_feature():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        from agent_framework._harness._agent import create_harness_agent
        # ... use experimental harness features
```

---

## 5 · AgentSession

**Module:** `agent_framework._sessions`
**Import:** `from agent_framework import AgentSession`

`AgentSession` is the lightweight state container passed through every agent run. It holds `session_id`, an optional `service_session_id` (provider-managed identifier), and a free-form `state` dict that all `ContextProvider` instances can read and write.

```python
Constructor:
AgentSession(
    *,
    session_id: str | None = None,           # auto-generated UUID if omitted
    service_session_id: str | ServiceSessionId | None = None,
)

Key attributes:
  session.session_id        → str (read-only property)
  session.service_session_id → str | ServiceSessionId | None
  session.state              → dict[str, Any]  (mutable)

Serialisation:
  session.to_dict()          → dict[str, Any]
  AgentSession.from_dict(d)  → AgentSession
```

**Example 13 — multi-turn conversation with shared session**

```python
import asyncio
from agent_framework import Agent, AgentSession

async def multi_turn_chat(agent: Agent) -> None:
    session = agent.create_session()
    print(f"Session ID: {session.session_id}")

    for user_message in [
        "My name is Alice.",
        "What is my name?",
    ]:
        response = await agent.run(user_message, session=session)
        print(f"User: {user_message}")
        print(f"Agent: {response.text}\n")

# asyncio.run(multi_turn_chat(your_agent))
```

**Example 14 — serialise and restore a session**

```python
import json
from agent_framework import AgentSession

# Store provider state in the session before serialising
session = AgentSession(session_id="conv-001")
session.state["user_name"] = "Alice"
session.state["turn_count"] = 3

# Persist
data = session.to_dict()
payload = json.dumps(data)

# Restore from storage (e.g. Redis, database)
restored = AgentSession.from_dict(json.loads(payload))
assert restored.session_id == "conv-001"
assert restored.state["user_name"] == "Alice"
print(f"Restored turn count: {restored.state['turn_count']}")
```

**Example 15 — service_session_id for provider-managed history**

```python
import asyncio
from agent_framework import Agent, AgentSession
from agent_framework.foundry import FoundryChatClient

async def main():
    client = FoundryChatClient(model="gpt-4o")
    agent = Agent(client=client, name="assistant", instructions="Be helpful.")

    # On first turn, session_id is generated; service_session_id may be set
    # by the provider (e.g. Azure AI Agents thread ID) during the run.
    session = AgentSession(session_id="user-42-thread")
    await agent.run("Hello!", session=session)

    # Provider may have stamped a service-managed ID
    if session.service_session_id:
        print(f"Provider session: {session.service_session_id}")

    # Pass the same session on subsequent turns to resume conversation
    response = await agent.run("What did I say before?", session=session)
    print(response.text)

# asyncio.run(main())
```

---

## 6 · AgentResponse · AgentResponseUpdate

**Module:** `agent_framework._sessions`
**Import:** `from agent_framework import AgentResponse, AgentResponseUpdate`

`AgentResponse` is the typed result of `await agent.run(...)`. It holds the assistant messages, structured output (if a response format was specified), usage details, and finish reason.

```python
Constructor:
AgentResponse(
    *,
    messages: Message | Sequence[Message] | None = None,
    response_id: str | None = None,
    agent_id: str | None = None,
    created_at: datetime | None = None,
    finish_reason: str | None = None,   # common values: "stop", "length", "tool_calls", "content_filter"
    usage_details: UsageDetails | None = None,
    value: ResponseModelT | None = None,       # structured output (Pydantic model)
    response_format: type[ResponseModelT] | None = None,
    additional_properties: dict[str, Any] | None = None,
)

Key properties:
  response.text            → str  (text of last assistant message)
  response.messages        → list[Message]
  response.usage_details   → UsageDetails | None
  response.finish_reason   → str | None
  response.value           → ResponseModelT | None  (structured output)

Class methods:
  AgentResponse.from_updates(updates: list[AgentResponseUpdate]) → AgentResponse
  AgentResponse.from_dict(d) → AgentResponse
  AgentResponse.from_json(s) → AgentResponse

Instance methods:
  response.to_dict()  → dict[str, Any]
  response.to_json()  → str
```

**Example 16 — inspect usage and finish_reason**

```python
import asyncio
from agent_framework import Agent

async def main(agent: Agent) -> None:
    response = await agent.run("Write a haiku about asyncio.")
    print("Text:", response.text)
    print("Finish reason:", response.finish_reason)
    if response.usage_details:
        print("Input tokens:", response.usage_details.get("input_token_count"))
        print("Output tokens:", response.usage_details.get("output_token_count"))

# asyncio.run(main(your_agent))
```

**Example 17 — structured output via `value`**

```python
import asyncio
from pydantic import BaseModel
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient

class TaskPlan(BaseModel):
    steps: list[str]
    estimated_minutes: int

async def main():
    client = FoundryChatClient(model="gpt-4o")
    agent = Agent(client=client, name="planner", instructions="Plan tasks as structured JSON.")

    response = await agent.run(
        "Plan a 3-step process to deploy a FastAPI service.",
        options={"response_format": TaskPlan},
    )
    plan: TaskPlan = response.value  # fully typed Pydantic model
    for i, step in enumerate(plan.steps, 1):
        print(f"Step {i}: {step}")
    print(f"ETA: {plan.estimated_minutes} minutes")

# asyncio.run(main())
```

**Example 18 — round-trip serialisation**

```python
import json, asyncio
from agent_framework import Agent, AgentResponse

async def persist_response(agent: Agent) -> None:
    response = await agent.run("What is the capital of France?")

    # Serialise for caching or logging
    payload = response.to_json()
    with open("response_cache.json", "w") as f:
        f.write(payload)

    # Restore from cache
    with open("response_cache.json") as f:
        restored = AgentResponse.from_json(f.read())

    assert restored.text == response.text
    print("Restored:", restored.text)

# asyncio.run(persist_response(your_agent))
```

---

## 7 · ChatContext

**Module:** `agent_framework._sessions`
**Import:** `from agent_framework import ChatContext, ChatMiddleware`

`ChatContext` is the mutable bag passed through the `ChatMiddleware` pipeline. Every middleware receives the same context object so it can inspect the outgoing request, augment it, intercept the result, or attach hooks that fire during streaming.

```python
Key attributes:
  context.client           → BaseChatClient   (the underlying LLM client)
  context.messages         → list[Message]    (mutable — prepend/append freely)
  context.options          → dict[str, Any]   (model, temperature, etc.)
  context.stream           → bool
  context.session          → AgentSession | None
  context.metadata         → dict[str, Any]   (shared across middleware in the pipeline)
  context.result           → ChatResponse | ResponseStream | None  (set after call_next)
  context.stream_transform_hooks  → list[Callable]  (per-update transforms)
  context.stream_result_hooks     → list[Callable]  (post-finalize transforms)
  context.stream_cleanup_hooks    → list[Callable]  (after stream consumed)
  context.function_invocation_kwargs → dict  (forwarded only to tool layers)
```

**Example 19 — latency and token logging middleware**

```python
import asyncio, time
from agent_framework import ChatMiddleware, ChatContext

class LatencyLoggingMiddleware(ChatMiddleware):
    async def process(self, context: ChatContext, call_next):
        start = time.monotonic()
        await call_next()
        elapsed_ms = (time.monotonic() - start) * 1000
        usage = getattr(context.result, "usage_details", {}) or {}
        print(
            f"[{context.client.__class__.__name__}] "
            f"model={context.options.get('model')} "
            f"elapsed={elapsed_ms:.0f}ms "
            f"tokens={usage.get('total_token_count', '?')}"
        )
```

**Example 20 — PII redaction via `stream_transform_hooks`**

```python
import re, asyncio
from agent_framework import ChatMiddleware, ChatContext, ChatResponseUpdate

class PIIRedactionMiddleware(ChatMiddleware):
    """Redact email addresses from every streaming chunk."""

    EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

    async def process(self, context: ChatContext, call_next):
        def redact_update(update: ChatResponseUpdate) -> ChatResponseUpdate:
            for content in update.contents or []:
                if content.type == "text":
                    content.text = self.EMAIL_RE.sub("[REDACTED]", content.text)
            return update

        context.stream_transform_hooks.append(redact_update)
        await call_next()
```

**Example 21 — short-circuit cache via result override**

```python
import asyncio
from agent_framework import ChatMiddleware, ChatContext, ChatResponse, Message

class SimpleCacheMiddleware(ChatMiddleware):
    _cache: dict[str, str] = {}

    async def process(self, context: ChatContext, call_next):
        # Build a simple cache key from the last user message
        last_user = next(
            (m for m in reversed(context.messages) if m.role == "user"), None
        )
        key = last_user.content_str if last_user else ""

        if key in self._cache and not context.stream:
            # Short-circuit non-streaming requests only; streaming requires ResponseStream
            context.result = ChatResponse(
                messages=[Message(role="assistant", contents=[self._cache[key]])],
                response_id="cached",
            )
            return

        await call_next()

        # Cache the response for next time
        if isinstance(context.result, ChatResponse):
            self._cache[key] = context.result.messages[-1].content_str
```

---

## 8 · ResponseStream

**Module:** `agent_framework._clients`
**Import:** `from agent_framework import ResponseStream`

`ResponseStream` is the async iterable wrapper returned when `stream=True`. It supports a `finalizer` (to produce the combined `ChatResponse` after draining), and four hook lists for intercepting data at different points in the stream lifecycle.

```python
Constructor:
ResponseStream(
    stream: AsyncIterable[UpdateT] | Awaitable[AsyncIterable[UpdateT]],
    *,
    finalizer: Callable[[Sequence[UpdateT]], FinalT | Awaitable[FinalT]] | None = None,
    transform_hooks: list[Callable[[UpdateT], UpdateT | Awaitable[UpdateT | None] | None]] | None = None,
    cleanup_hooks: list[Callable[[], Awaitable[None] | None]] | None = None,
    result_hooks: list[Callable[[FinalT], FinalT | Awaitable[FinalT | None] | None]] | None = None,
)

Key methods:
  async for update in stream:  ...              # iterate updates
  final = await stream.get_final_response()     # drain + run finalizer → FinalT
  stream.with_result_hook(fn)                   # attach a post-finalizer transform hook
  stream.with_transform_hook(fn)                # attach a per-update transform hook
  stream.with_cleanup_hook(fn)                  # attach a post-consumption cleanup hook
  stream.with_finalizer(fn)                     # replace the finalizer (returns new stream)
```

**Example 22 — consume streaming response token-by-token**

```python
import asyncio
from agent_framework import Agent

async def stream_to_console(agent: Agent, prompt: str) -> None:
    stream = agent.run(prompt, stream=True)
    async for update in stream:
        print(update.text, end="", flush=True)
    print()  # newline after stream ends
    response = await stream.get_final_response()
    print(f"\n[done — finish_reason: {response.finish_reason}]")

# asyncio.run(stream_to_console(your_agent, "Explain asyncio in 3 points."))
```

**Example 23 — attach a `result_hook` to log final usage**

```python
import asyncio
from agent_framework import Agent, AgentResponse

async def main(agent: Agent) -> None:
    stream = agent.run("What is machine learning?", stream=True)

    async def log_usage(response: AgentResponse) -> AgentResponse:
        if response.usage_details:
            print(f"Final usage: {response.usage_details}")
        return response

    stream.with_result_hook(log_usage)

    async for _ in stream:
        pass  # consume
    await stream.get_final_response()

# asyncio.run(main(your_agent))
```

**Example 24 — build a custom `ResponseStream` for testing**

```python
import asyncio
from agent_framework import ResponseStream, ChatResponseUpdate, ChatResponse, Message, Content

async def fake_stream_source():
    for word in ["Hello", " from", " a", " test", " stream"]:
        yield ChatResponseUpdate(role="assistant", contents=[Content(type="text", text=word)])

def finalizer(updates):
    full_text = "".join(
        c.text
        for u in updates
        for c in (u.contents or [])
        if c.type == "text"
    )
    return ChatResponse(
        messages=[Message(role="assistant", contents=[full_text])],
        response_id="test-stream",
    )

async def main():
    stream = ResponseStream(fake_stream_source(), finalizer=finalizer)
    chunks = []
    async for update in stream:
        chunks.append(update)
    response = await stream.get_final_response()
    print(response.messages[0].content_str)  # "Hello from a test stream"

asyncio.run(main())
```

---

## 9 · BaseEmbeddingClient · GeneratedEmbeddings · EmbeddingGenerationOptions

**Module:** `agent_framework._clients`
**Import:** `from agent_framework import BaseEmbeddingClient, GeneratedEmbeddings, EmbeddingGenerationOptions`

These classes define the embedding interface used by context providers (e.g. `AzureAISearchContextProvider`, `Mem0ContextProvider`) and allow you to plug in any embedding backend.

### `EmbeddingGenerationOptions` (TypedDict, total=False)

```python
class EmbeddingGenerationOptions(TypedDict, total=False):
    model: str        # e.g. "text-embedding-3-small"
    dimensions: int   # e.g. 1536
```

### `GeneratedEmbeddings`

Extends `list[Embedding[T]]` so you can index and iterate directly.

```python
Constructor:
GeneratedEmbeddings(
    embeddings: Iterable[Embedding[T]] | None = None,
    *,
    options: EmbeddingOptionsT | None = None,
    usage: UsageDetails | None = None,
    additional_properties: dict[str, Any] | None = None,
)
# Embedding is a dataclass with .vector: list[float]
```

### `BaseEmbeddingClient` (ABC)

```python
Constructor:
BaseEmbeddingClient(
    *,
    additional_properties: dict[str, Any] | None = None,
)

Abstract method (must implement):
  async def get_embeddings(
      self,
      values: Sequence[EmbeddingInputT],
      *,
      options: EmbeddingOptionsT | None = None,
  ) -> GeneratedEmbeddings[EmbeddingT, EmbeddingOptionsT]: ...

Class var:
  OTEL_PROVIDER_NAME: ClassVar[str] = "unknown"  # override for telemetry
```

**Example 25 — custom sentence-transformers embedding client**

```python
import asyncio
from collections.abc import Sequence
from agent_framework import BaseEmbeddingClient, Embedding, GeneratedEmbeddings, EmbeddingGenerationOptions

class SentenceTransformersClient(BaseEmbeddingClient):
    """Wraps sentence-transformers for local embeddings."""

    OTEL_PROVIDER_NAME = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs) -> None:
        super().__init__(**kwargs)
        # Lazy import to avoid hard dependency
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    async def get_embeddings(
        self,
        values: Sequence[str],
        *,
        options: EmbeddingGenerationOptions | None = None,
    ) -> GeneratedEmbeddings:
        import asyncio
        loop = asyncio.get_running_loop()
        # Run CPU-bound model in a thread pool
        vectors = await loop.run_in_executor(
            None, lambda: self._model.encode(list(values), convert_to_numpy=True).tolist()
        )
        embeddings = [Embedding(vector=v) for v in vectors]
        return GeneratedEmbeddings(
            embeddings,
            options=options,
            usage={"prompt_tokens": sum(len(v.split()) for v in values)},
        )

async def main():
    client = SentenceTransformersClient()
    result = await client.get_embeddings(["Hello world", "Agent framework"])
    print(f"Got {len(result)} embeddings, dim={len(result[0].vector)}")

# asyncio.run(main())
```

**Example 26 — use `GeneratedEmbeddings` metadata**

```python
import asyncio
from agent_framework import GeneratedEmbeddings, Embedding

async def main():
    embeddings = GeneratedEmbeddings(
        [Embedding(vector=[0.1, 0.2, 0.3]), Embedding(vector=[0.4, 0.5, 0.6])],
        usage={"prompt_tokens": 8, "total_tokens": 8},
    )

    print(f"Count: {len(embeddings)}")                 # 2
    print(f"First vector: {embeddings[0].vector}")     # [0.1, 0.2, 0.3]
    print(f"Usage: {embeddings.usage}")                # {'prompt_tokens': 8, 'total_tokens': 8}

    # Iterate like a list
    for emb in embeddings:
        print(f"  dim={len(emb.vector)}")

asyncio.run(main())
```

**Example 27 — wire into a context provider**

```python
import asyncio
from agent_framework import Agent
from agent_framework.azure import AzureAISearchContextProvider

# Any BaseEmbeddingClient works here — swap SentenceTransformersClient with
# OpenAIEmbeddingClient, OllamaEmbeddingClient, or your custom class
async def main(chat_client, embedding_client, azure_search_endpoint, index_name):
    provider = AzureAISearchContextProvider(
        endpoint=azure_search_endpoint,
        index_name=index_name,
        embedding_function=embedding_client,
        vector_field_name="content_vector",
        top_k=5,
    )
    agent = Agent(
        client=chat_client,
        name="rag-agent",
        instructions="Answer questions using the provided context.",
        context_providers=[provider],
    )
    response = await agent.run("What is the refund policy?")
    print(response.text)

# asyncio.run(main(your_chat_client, your_embedding_client, "https://...", "docs-index"))
```

---

## 10 · BaseChatClient

**Module:** `agent_framework._clients`
**Import:** `from agent_framework import BaseChatClient`

`BaseChatClient` is the abstract base for building custom LLM providers. Subclass it to add support for any API not covered by the built-in clients (OpenAI, Anthropic, Bedrock, Ollama, Foundry, Vertex).

```python
Constructor:
BaseChatClient(
    *,
    additional_properties: dict[str, Any] | None = None,
)

Abstract method (must implement):
  def _inner_get_response(       # must be a regular def, NOT async def
      self,
      *,
      messages: list[Message],
      stream: bool,
      options: dict[str, Any],
      **kwargs,
  ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
      # stream=False → return a coroutine/awaitable (nest async logic in inner async def)
      # stream=True  → return self._build_response_stream(async_gen()) directly

Public API (inherited):
  await client.get_response(messages, *, stream=False, options=None, **kwargs)
    → ChatResponse (stream=False) | ResponseStream (stream=True)
  client.to_dict()   → dict[str, Any]
  BaseChatClient.from_dict(d)  → BaseChatClient   (requires REGISTRY entry)
```

**Example 28 — minimal custom provider (non-streaming)**

```python
import asyncio
from collections.abc import Awaitable
from agent_framework import BaseChatClient, ChatResponse, ChatResponseUpdate as Update, Message, ResponseStream

class EchoClient(BaseChatClient):
    """Trivial client that echoes the last user message — useful for testing."""

    def _inner_get_response(
        self,
        *,
        messages: list[Message],
        stream: bool,
        options: dict,
        **kwargs,
    ) -> Awaitable[ChatResponse] | ResponseStream[Update, ChatResponse]:
        last_user = next(
            (m.content_str for m in reversed(messages) if m.role == "user"),
            "(no user message)",
        )
        reply = f"Echo: {last_user}"

        if stream:
            async def _stream():
                yield Update(role="assistant", contents=[{"type": "text", "text": reply}])
            return self._build_response_stream(_stream())
        else:
            async def _response():
                return ChatResponse(
                    messages=[Message(role="assistant", contents=[reply])],
                    response_id="echo-001",
                )
            return _response()

async def main():
    from agent_framework import Agent
    agent = Agent(client=EchoClient(), name="echo", instructions="Echo the user.")
    response = await agent.run("Hello!")
    print(response.text)  # "Echo: Hello!"

asyncio.run(main())
```

**Example 29 — streaming provider with model routing**

```python
import asyncio, httpx, json
from collections.abc import Awaitable
from agent_framework import BaseChatClient, ChatResponse, ChatResponseUpdate, Message, ResponseStream

class CustomLLMClient(BaseChatClient):
    """Calls a self-hosted OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, model: str = "mistral-7b", **kwargs) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._model = model

    def _inner_get_response(
        self,
        *,
        messages: list[Message],
        stream: bool,
        options: dict,
        **kwargs,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        payload = {
            "model": options.get("model", self._model),
            "messages": [{"role": m.role, "content": m.content_str} for m in messages],
            "stream": stream,
        }
        if stream:
            async def _stream():
                async with httpx.AsyncClient() as http:
                    async with http.stream("POST", f"{self._base_url}/v1/chat/completions", json=payload) as r:
                        r.raise_for_status()
                        async for line in r.aiter_lines():
                            if line.startswith("data: ") and line != "data: [DONE]":
                                chunk = json.loads(line[6:])
                                delta = chunk["choices"][0]["delta"]
                                if text := delta.get("content", ""):
                                    yield ChatResponseUpdate(
                                        role="assistant",
                                        contents=[{"type": "text", "text": text}],
                                    )
            return self._build_response_stream(_stream())
        else:
            async def _response():
                async with httpx.AsyncClient() as http:
                    r = await http.post(f"{self._base_url}/v1/chat/completions", json=payload)
                    r.raise_for_status()
                    data = r.json()
                text = data["choices"][0]["message"]["content"]
                return ChatResponse(
                    messages=[Message(role="assistant", contents=[text])],
                    response_id=data.get("id"),
                )
            return _response()

# asyncio.run(Agent(client=CustomLLMClient("http://localhost:8000"), ...).run("Hello!"))
```

**Example 30 — `to_dict` / `from_dict` for config serialisation**

```python
from agent_framework import BaseChatClient

class ConfigurableClient(BaseChatClient):
    DEFAULT_EXCLUDE = {"_api_key"}  # never serialise the key

    def __init__(self, endpoint: str, api_key: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.endpoint = endpoint
        self._api_key = api_key   # excluded via DEFAULT_EXCLUDE

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        raise NotImplementedError

    def to_dict(self):
        base = super().to_dict()
        base["endpoint"] = self.endpoint
        return base

client = ConfigurableClient(endpoint="https://my-llm.example.com", api_key="secret")
d = client.to_dict()
print(d["endpoint"])    # "https://my-llm.example.com"
print("_api_key" not in d)  # True — excluded
```

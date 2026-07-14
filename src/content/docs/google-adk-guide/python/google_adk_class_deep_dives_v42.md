---
title: "Class deep dives — volume 42 (Gemini model class, BaseLlmConnection, MCP transports, BaseCriterion/EvalMetric, McpTool/ProgressCallbackFactory, EvalCase/Invocation, ResponseEvaluator, BaseSessionService/GetSessionConfig, ArtifactVersion/BaseArtifactService, EvalSet)"
description: "10 source-verified deep dives for google-adk 2.4.0: Gemini model class (use_interactions_api, retry_options, speech_config, api_client override), BaseLlmConnection live-streaming ABC (send_history/send_content/send_realtime/receive/close), SseConnectionParams+StreamableHTTPConnectionParams+StdioConnectionParams MCP transports, BaseCriterion+EvalMetric+JudgeModelOptions eval metric configuration, McpTool+ProgressCallbackFactory per-tool progress callbacks, EvalCase+Invocation+IntermediateData core eval data model, ResponseEvaluator (coherence + ROUGE match), BaseSessionService+GetSessionConfig+ListSessionsResponse session service ABC, ArtifactVersion+BaseArtifactService artifact metadata ABC, EvalSet eval case container."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 42"
  order: 111
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source (locate yours with
`python -c 'import google.adk; print(google.adk.__file__)'`) on
**google-adk == 2.4.0**. No documentation or blog posts were used as primary
sources.
</Aside>

---

## 1 · `Gemini` — first-party Gemini model class

**Source:** `google/adk/models/google_llm.py`

`Gemini` is the concrete `BaseLlm` for all Google Gemini models. When you
pass a model string such as `"gemini-2.5-flash"` to `LlmAgent(model=...)`, the
`LLMRegistry` resolves it to `Gemini`. Constructing `Gemini` directly unlocks
advanced fields: automatic HTTP retries, the Interactions API for stateful
server-side history, a custom TTS speech config, and full control over the
underlying `google.genai.Client`.

### Constructor signature (verified `google_llm.py`)

```python
class Gemini(BaseLlm):
    model: str = "gemini-2.5-flash"

    # Extra kwargs forwarded verbatim to google.genai.Client()
    client_kwargs: Optional[dict[str, Any]] = None

    # Override the AI platform base URL (e.g. Vertex AI regional endpoints)
    base_url: Optional[str] = None

    # TTS voice/encoding for Live-mode sessions
    speech_config: Optional[types.SpeechConfig] = None

    # Use client.aio.interactions instead of generate_content
    use_interactions_api: bool = False

    # HTTP-level retry policy (google.genai.types.HttpRetryOptions)
    retry_options: Optional[types.HttpRetryOptions] = None
```

Override `api_client` (`@cached_property`) in a subclass to supply any
`google.genai.Client` constructor argument ADK does not expose as a field:
`project`, `location`, `credentials`, `http_options`, `enterprise`, etc.

### Example 1 — retry-aware Gemini with exponential back-off

```python
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.genai import types

resilient_model = Gemini(
    model="gemini-2.5-flash",
    retry_options=types.HttpRetryOptions(
        attempts=4,
        initial_delay=1.0,
        exp_base=2.0,
        max_delay=30.0,
    ),
)

agent = LlmAgent(
    name="resilient_agent",
    model=resilient_model,
    instruction="You are a resilient assistant.",
)
```

### Example 2 — Interactions API for server-managed conversation history

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

# Interactions API keeps conversation history on the server side.
# ADK converts responses to the standard LlmResponse format transparently.
interactions_model = Gemini(
    model="gemini-2.5-flash",
    use_interactions_api=True,
)

agent = LlmAgent(
    name="stateful_agent",
    model=interactions_model,
    instruction="Assist the user. Remember what they said.",
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="stateful_app")
    session = await runner.session_service.create_session(
        app_name="stateful_app", user_id="u1"
    )
    for turn in ["Hello, my name is Alice.", "What is my name?"]:
        async for event in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part(text=turn)]
            ),
        ):
            if event.is_final_response() and event.content:
                print(f"[{turn!r}] →", event.content.parts[0].text)
    await runner.close()

asyncio.run(main())
```

### Example 3 — custom `api_client` for a regional Vertex AI endpoint

```python
from functools import cached_property
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.genai import Client


class RegionalGemini(Gemini):
    """Pins all requests to a specific Vertex AI region."""

    @cached_property
    def api_client(self) -> Client:
        return Client(
            vertexai=True,
            project="my-gcp-project",
            location="europe-west4",
        )


agent = LlmAgent(
    name="regional_agent",
    model=RegionalGemini(model="gemini-2.5-pro"),
    instruction="You run on a regional Vertex AI endpoint.",
)
```

---

## 2 · `BaseLlmConnection` — live streaming connection ABC

**Source:** `google/adk/models/base_llm_connection.py`

`BaseLlmConnection` is the abstract base class for bidirectional, real-time LLM
connections used in `LiveRequestQueue`-based live sessions. The concrete
implementation is `GeminiLlmConnection`. `BaseLlmConnection` defines the
contract any custom live adapter must satisfy.

### Abstract method surface (verified `base_llm_connection.py`)

| Method | Direction | Description |
|--------|-----------|-------------|
| `send_history(history)` | client → model | Push full conversation history right after setup |
| `send_content(content)` | client → model | Send a turn-completing user `Content`; model replies immediately |
| `_send_content(content, *, partial=False)` | client → model | Partial-update hook; default delegates to `send_content` |
| `send_realtime(blob)` | client → model | Push raw audio/video chunk; VAD decides when to respond |
| `receive()` | model → client | `AsyncGenerator[LlmResponse, None]` — yields model response events |
| `close()` | — | Tear down the transport connection |

### Example 1 — minimal echo connection for unit testing

```python
from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from google.genai import types
from google.adk.models.base_llm_connection import BaseLlmConnection
from google.adk.models.llm_response import LlmResponse


class EchoConnection(BaseLlmConnection):
    """Toy connection that echoes every sent content as a model reply."""

    def __init__(self):
        self._queue: asyncio.Queue[LlmResponse] = asyncio.Queue()

    async def send_history(self, history: list[types.Content]) -> None:
        pass  # No-op: history not needed for echo

    async def send_content(self, content: types.Content) -> None:
        echo_text = " ".join(
            p.text for p in (content.parts or []) if p.text
        )
        await self._queue.put(
            LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"Echo: {echo_text}")],
                )
            )
        )

    async def send_realtime(self, blob: types.Blob) -> None:
        await self._queue.put(
            LlmResponse(content=types.Content(role="model", parts=[]))
        )

    async def receive(self) -> AsyncGenerator[LlmResponse, None]:  # type: ignore[override]
        while True:
            response = await self._queue.get()
            yield response
            if response.content and response.content.parts:
                break  # Emit one complete turn then stop

    async def close(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()


async def demo():
    conn = EchoConnection()
    await conn.send_content(
        types.Content(role="user", parts=[types.Part(text="Hello!")])
    )
    async for resp in conn.receive():
        print(resp.content.parts[0].text)  # → Echo: Hello!
        break

asyncio.run(demo())
```

### Example 2 — `_send_content` partial-update hook

```python
from google.genai import types
from google.adk.models.base_llm_connection import BaseLlmConnection
from google.adk.models.llm_response import LlmResponse


class PartialAwareConnection(BaseLlmConnection):
    """Records whether each chunk is a partial or turn-completing update."""

    def __init__(self):
        self._log: list[str] = []

    async def send_history(self, history): pass
    async def send_realtime(self, blob): pass

    async def send_content(self, content: types.Content) -> None:
        text = " ".join(p.text for p in (content.parts or []) if p.text)
        self._log.append(f"[complete] {text}")

    async def _send_content(
        self, content: types.Content, *, partial: bool = False
    ) -> None:
        if partial:
            text = " ".join(p.text for p in (content.parts or []) if p.text)
            self._log.append(f"[partial] {text}")
        else:
            await self.send_content(content)

    async def receive(self): # type: ignore[override]
        yield LlmResponse()

    async def close(self): pass


async def show():
    conn = PartialAwareConnection()
    await conn._send_content(
        types.Content(role="user", parts=[types.Part(text="draft")]),
        partial=True,
    )
    await conn._send_content(
        types.Content(role="user", parts=[types.Part(text="final")]),
        partial=False,
    )
    print(conn._log)
    # ['[partial] draft', '[complete] final']

import asyncio
asyncio.run(show())
```

### Example 3 — safe generator teardown with `Aclosing`

```python
from google.adk.utils.context_utils import Aclosing
from google.adk.models.base_llm_connection import BaseLlmConnection

async def drain_connection(conn: BaseLlmConnection) -> list[str]:
    """Collect all text chunks, closing the generator on exit."""
    texts: list[str] = []
    async with Aclosing(conn.receive()) as gen:
        async for resp in gen:
            if resp.content and resp.content.parts:
                for part in resp.content.parts:
                    if part.text:
                        texts.append(part.text)
    return texts
```

`Aclosing` (from `google.adk.utils.context_utils`) ensures `aclose()` is
called on the async generator even if the consumer breaks early — matching the
pattern used in `GeminiLlmConnection.receive()`.

---

## 3 · `SseConnectionParams` + `StreamableHTTPConnectionParams` + `StdioConnectionParams` — MCP transport configurations

**Source:** `google/adk/tools/mcp_tool/mcp_session_manager.py`

These three Pydantic models are the transport-level connection descriptors fed
to `McpToolset`. Choose based on the protocol your MCP server speaks.

### Field reference (verified `mcp_session_manager.py`)

```python
class StdioConnectionParams(BaseModel):
    server_params: StdioServerParameters   # from mcp.StdioServerParameters
    timeout: float = 5.0                   # connect timeout in seconds

class SseConnectionParams(BaseModel):
    url: str
    headers: dict[str, Any] | None = None
    timeout: float = 5.0
    sse_read_timeout: float = 300.0        # 5-minute default
    httpx_client_factory: CheckableMcpHttpClientFactory = create_mcp_http_client

class StreamableHTTPConnectionParams(BaseModel):
    url: str
    headers: dict[str, Any] | None = None
    timeout: float = 5.0
    sse_read_timeout: float = 300.0
    terminate_on_close: bool = True        # shut down server on disconnect
    httpx_client_factory: CheckableMcpHttpClientFactory = create_mcp_http_client
```

**`CheckableMcpHttpClientFactory`** is a `@runtime_checkable Protocol`. Inject
it to provide a custom HTTPX client (e.g. private CA trust, proxy settings, or
a connection pool).

### Example 1 — `StdioConnectionParams` for a local subprocess MCP server

```python
from mcp import StdioServerParameters
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "my_mcp_server"],
            env={"LOG_LEVEL": "info"},
        ),
        timeout=10.0,
    )
)

agent = LlmAgent(
    name="local_mcp_agent",
    model="gemini-2.5-flash",
    tools=[toolset],
)
```

### Example 2 — `SseConnectionParams` with auth headers and extended read timeout

```python
import os
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

toolset = McpToolset(
    connection_params=SseConnectionParams(
        url="https://mcp.example.com/sse",
        headers={"Authorization": f"Bearer {os.environ['MCP_TOKEN']}"},
        timeout=5.0,
        sse_read_timeout=600.0,  # 10-minute long-running operations
    )
)
```

### Example 3 — `StreamableHTTPConnectionParams` with a custom HTTPX factory

```python
import os
import httpx
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StreamableHTTPConnectionParams,
)
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset


def private_ca_factory(headers=None, timeout=None, **_) -> httpx.AsyncClient:
    """Returns an AsyncClient trusting a private CA certificate.

    ADK passes headers= and timeout= from MCPSessionManager._create_client;
    accept them explicitly so our own timeout wins without a duplicate-kwarg error.
    """
    return httpx.AsyncClient(
        verify="/etc/ssl/private-ca.pem",
        timeout=httpx.Timeout(60.0),   # override framework timeout
        headers=headers or {},
    )


toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://internal-mcp.corp.example.com/mcp",
        headers={"X-Service-Token": os.environ["SERVICE_TOKEN"]},
        terminate_on_close=True,
        httpx_client_factory=private_ca_factory,
    )
)
```

---

## 4 · `BaseCriterion` + `EvalMetric` + `JudgeModelOptions` — evaluation metric configuration

**Source:** `google/adk/evaluation/eval_metrics.py`

`EvalMetric` is the top-level descriptor that wires a metric name, a threshold,
and a `BaseCriterion` together. `JudgeModelOptions` controls the LLM that acts
as the judge. These types are the building blocks for every ADK evaluator.

### Class signatures (verified `eval_metrics.py`)

```python
class EvalStatus(Enum):
    PASSED = 1
    FAILED = 2
    NOT_EVALUATED = 3

class JudgeModelOptions(EvalBaseModel):
    judge_model: str = "gemini-2.5-flash"
    judge_model_config: Optional[genai_types.GenerateContentConfig] = None
    num_samples: int = 5                   # sampling count per invocation

class BaseCriterion(BaseModel):
    # camelCase JSON aliases (alias_generator=to_camel)
    threshold: float                       # required pass/fail boundary
    include_intermediate_responses_in_final: bool = False

class LlmAsAJudgeCriterion(BaseCriterion):
    judge_model_options: JudgeModelOptions = JudgeModelOptions()

class RubricsBasedCriterion(BaseCriterion):
    judge_model_options: JudgeModelOptions = JudgeModelOptions()
    rubrics: list[Rubric] = []             # Rubric(rubric_id, rubric_content) objects

class LlmBackedUserSimulatorCriterion(BaseCriterion):
    pass

class EvalMetric(EvalBaseModel):
    metric_name: str                       # PrebuiltMetrics value or custom name
    threshold: float
    criterion: Optional[BaseCriterion] = None
    custom_function_path: Optional[str] = None
```

### `PrebuiltMetrics` values (verified `eval_metrics.py`)

```python
class PrebuiltMetrics(Enum):
    TOOL_TRAJECTORY_AVG_SCORE                = "tool_trajectory_avg_score"
    RESPONSE_EVALUATION_SCORE                = "response_evaluation_score"
    RESPONSE_MATCH_SCORE                     = "response_match_score"
    SAFETY_V1                                = "safety_v1"
    FINAL_RESPONSE_MATCH_V2                  = "final_response_match_v2"
    RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1   = "rubric_based_final_response_quality_v1"
    HALLUCINATIONS_V1                        = "hallucinations_v1"
    RUBRIC_BASED_TOOL_USE_QUALITY_V1         = "rubric_based_tool_use_quality_v1"
    MULTI_TURN_TASK_SUCCESS_V1               = "multi_turn_task_success_v1"
    MULTI_TURN_TRAJECTORY_QUALITY_V1         = "multi_turn_trajectory_quality_v1"
    MULTI_TURN_TOOL_USE_QUALITY_V1           = "multi_turn_tool_use_quality_v1"
```

### Example 1 — building an `EvalMetric` with a rubrics-based criterion

```python
from google.adk.evaluation.eval_metrics import (
    EvalMetric, JudgeModelOptions, RubricsBasedCriterion,
    PrebuiltMetrics,
)
from google.adk.evaluation.eval_rubrics import Rubric, RubricContent
from google.genai import types as genai_types

metric = EvalMetric(
    # RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1 dispatches to the rubric scorer,
    # which reads criterion.rubrics; RESPONSE_EVALUATION_SCORE would ignore them.
    metric_name=PrebuiltMetrics.RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1.value,
    threshold=0.7,  # rubric confidence score range is 0–1
    criterion=RubricsBasedCriterion(
        threshold=0.7,
        rubrics=[
            Rubric(rubric_id="r1", rubric_content=RubricContent(text_property="Is the response factually correct?")),
            Rubric(rubric_id="r2", rubric_content=RubricContent(text_property="Is the response concise and well-structured?")),
            Rubric(rubric_id="r3", rubric_content=RubricContent(text_property="Does the response fully address the user's question?")),
        ],
        judge_model_options=JudgeModelOptions(
            judge_model="gemini-2.5-flash",
            num_samples=3,
            judge_model_config=genai_types.GenerateContentConfig(
                temperature=0.0,
            ),
        ),
    ),
)
print(metric.metric_name)      # rubric_based_final_response_quality_v1
print(metric.threshold)        # 0.7
```

### Example 2 — ROUGE match metric (no judge model needed)

```python
from google.adk.evaluation.eval_metrics import EvalMetric, PrebuiltMetrics

rouge_metric = EvalMetric(
    metric_name=PrebuiltMetrics.RESPONSE_MATCH_SCORE.value,
    threshold=0.5,   # ROUGE-1 F-score ≥ 0.5 → PASSED
)
print(rouge_metric.metric_name)  # response_match_score
```

### Example 3 — serialising `EvalMetric` to JSON (camelCase aliases)

```python
from google.adk.evaluation.eval_metrics import EvalMetric, RubricsBasedCriterion
from google.adk.evaluation.eval_rubrics import Rubric, RubricContent

metric = EvalMetric(
    metric_name="custom_accuracy",
    threshold=0.8,
    criterion=RubricsBasedCriterion(
        threshold=0.8,
        rubrics=[
            Rubric(rubric_id="r1", rubric_content=RubricContent(text_property="The answer is numerically accurate.")),
        ],
    ),
)

# Pydantic model_dump with camelCase (alias_generator=to_camel in BaseCriterion)
raw = metric.model_dump_json(by_alias=True, indent=2)
restored = EvalMetric.model_validate_json(raw)
assert restored.metric_name == "custom_accuracy"
print("Round-trip OK")
```

---

## 5 · `McpTool` + `ProgressCallbackFactory` — individual MCP tool and per-tool progress callbacks

**Source:** `google/adk/tools/mcp_tool/mcp_tool.py`

`McpTool` (alias `MCPTool`) is the wrapper `BaseTool` returned by
`McpToolset.get_tools()`. Each instance wraps one `mcp.types.Tool` entry and
handles: JSON Schema → Gemini `FunctionDeclaration` conversion, auth injection
via `BaseAuthenticatedTool`, graceful error boundary on transport crashes, and
optional progress reporting via `ProgressCallbackFactory`.

### `ProgressCallbackFactory` Protocol (verified `mcp_tool.py`)

```python
@runtime_checkable
class ProgressCallbackFactory(Protocol):
    def __call__(
        self,
        tool_name: str,
        *,
        callback_context: CallbackContext | None = None,
        **kwargs: Any,          # required for forward compatibility
    ) -> ProgressFnT | None: ...
```

The factory receives the tool name and a `CallbackContext` (write access to
session state) and returns a `ProgressFnT` coroutine or `None` to suppress
reporting for that tool.

### Example 1 — progress factory that writes to session state

```python
from mcp.shared.session import ProgressFnT
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset


def session_progress_factory(
    tool_name: str,
    *,
    callback_context: CallbackContext | None = None,
    **kwargs,
) -> ProgressFnT | None:
    async def on_progress(progress: float, total: float | None, message: str | None):
        pct = f"{progress/total*100:.0f}%" if total else f"{progress}"
        print(f"[{tool_name}] {pct} — {message}")
        if callback_context:
            callback_context.state[f"{tool_name}_progress"] = pct
    return on_progress


toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://mcp.example.com/mcp",
    ),
    progress_callback=session_progress_factory,
)
```

### Example 2 — inspecting a `McpTool` instance at runtime

```python
import asyncio
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_tool import McpTool


async def inspect_mcp_tools():
    toolset = McpToolset(
        connection_params=SseConnectionParams(url="https://mcp.example.com/sse"),
    )
    tools = await toolset.get_tools()
    for tool in tools:
        assert isinstance(tool, McpTool)
        print(f"name={tool.name}")
        print(f"description={tool.description[:60]!r}")
        print(f"is_long_running={tool.is_long_running}")
        print(f"custom_metadata={tool.custom_metadata}")
    await toolset.close()

asyncio.run(inspect_mcp_tools())
```

### Example 3 — selective progress suppression by tool name

```python
SILENT_TOOLS = {"slow_export", "batch_process"}


def selective_progress_factory(
    tool_name: str,
    *,
    callback_context=None,
    **kwargs,
):
    if tool_name in SILENT_TOOLS:
        return None  # No progress feedback for these noisy operations

    async def on_progress(progress, total, message):
        print(f"[{tool_name}] {progress}/{total}: {message}")

    return on_progress
```

---

## 6 · `EvalCase` + `Invocation` + `IntermediateData` — core evaluation data model

**Source:** `google/adk/evaluation/eval_case.py`

`EvalCase` is the fundamental unit in the ADK evaluation framework. Each case
represents one or more conversation turns (`Invocation`s) paired with expected
responses and optional rubrics. `IntermediateData` captures the full tool-use
trajectory alongside each response.

### Class hierarchy (verified `eval_case.py`)

```
EvalBaseModel (Pydantic)
├── IntermediateData    — tool_uses, tool_responses, intermediate_responses
├── InvocationEvent     — author, content
├── InvocationEvents    — list[InvocationEvent]
├── Invocation          — user_content, final_response, intermediate_data
├── SessionInput        — app_name, user_id, state
└── EvalCase            — eval_id, conversation, session_input, rubric
```

### `Invocation` fields (verified `eval_case.py`)

```python
class Invocation(EvalBaseModel):
    invocation_id: str = ""
    user_content: genai_types.Content          # required
    final_response: Optional[genai_types.Content] = None
    intermediate_data: Optional[Union[IntermediateData, InvocationEvents]] = None
    creation_timestamp: float = 0.0
```

### `IntermediateData` fields (verified `eval_case.py`)

```python
class IntermediateData(EvalBaseModel):
    tool_uses: list[genai_types.FunctionCall] = []
    tool_responses: list[genai_types.FunctionResponse] = []
    intermediate_responses: list[tuple[str, list[genai_types.Part]]] = []
    # tuple: (author_name, [Part, ...])
```

### Example 1 — single-turn `EvalCase` with tool trajectory

```python
from google.genai import types
from google.adk.evaluation.eval_case import EvalCase, IntermediateData, Invocation

case = EvalCase(
    eval_id="order_query_001",
    conversation=[
        Invocation(
            user_content=types.Content(
                role="user",
                parts=[types.Part(text="What is the status of order #1234?")],
            ),
            final_response=types.Content(
                role="model",
                parts=[types.Part(text="Order #1234 is shipped; arrives tomorrow.")],
            ),
            intermediate_data=IntermediateData(
                tool_uses=[
                    types.FunctionCall(
                        name="get_order_status", args={"order_id": "1234"}
                    )
                ],
                tool_responses=[
                    types.FunctionResponse(
                        name="get_order_status",
                        response={"status": "shipped", "eta": "tomorrow"},
                    )
                ],
            ),
        )
    ],
)
print(case.eval_id)  # order_query_001
```

### Example 2 — multi-turn `EvalCase` with `InvocationEvents` for sub-agent traces

```python
from google.adk.evaluation.eval_case import (
    EvalCase, Invocation, InvocationEvent, InvocationEvents,
)
from google.genai import types

case = EvalCase(
    eval_id="flight_booking_001",
    conversation=[
        Invocation(
            user_content=types.Content(
                role="user", parts=[types.Part(text="Book a flight to Paris")]
            ),
            final_response=types.Content(
                role="model",
                parts=[types.Part(text="Found a flight on Tuesday for $450. Shall I book?")],
            ),
            # InvocationEvents records sub-agent progress messages
            intermediate_data=InvocationEvents(
                invocation_events=[
                    InvocationEvent(
                        author="flight_search_agent",
                        content=types.Content(
                            role="model",
                            parts=[types.Part(text="Searching available flights…")],
                        ),
                    )
                ]
            ),
        ),
        Invocation(
            user_content=types.Content(
                role="user", parts=[types.Part(text="Yes, book it.")]
            ),
            final_response=types.Content(
                role="model",
                parts=[types.Part(text="Booked! Confirmation #PQ789.")],
            ),
        ),
    ],
)
print(len(case.conversation))  # 2
```

### Example 3 — JSON round-trip for `EvalCase`

```python
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.genai import types

case = EvalCase(
    eval_id="round_trip_test",
    conversation=[
        Invocation(
            user_content=types.Content(
                role="user", parts=[types.Part(text="Ping")]
            ),
            final_response=types.Content(
                role="model", parts=[types.Part(text="Pong")]
            ),
        )
    ],
)

json_str = case.model_dump_json(indent=2)
restored = EvalCase.model_validate_json(json_str)
assert restored.eval_id == case.eval_id
print("Round-trip OK:", restored.eval_id)
```

---

## 7 · `ResponseEvaluator` — concrete response quality evaluator

**Source:** `google/adk/evaluation/response_evaluator.py`

`ResponseEvaluator` is a concrete `Evaluator` that supports two built-in
metrics: **coherence** (scored 1–5 via Vertex AI) and **ROUGE-1 match**
(scored 0–1 against a golden response).

### Constructor (verified `response_evaluator.py`)

```python
class ResponseEvaluator(Evaluator):
    def __init__(
        self,
        threshold: Optional[float] = None,
        metric_name: Optional[str] = None,
        eval_metric: Optional[EvalMetric] = None,
    ): ...
```

Pass either `eval_metric` **or** both `threshold` + `metric_name` — mixing
raises `ValueError`. Supported `metric_name` values:

| Value | Backend | Range |
|-------|---------|-------|
| `"response_evaluation_score"` | Vertex AI `PrebuiltMetric.COHERENCE` | 1–5 |
| `"response_match_score"` | ROUGE-1 F-measure | 0–1 |

### Example 1 — ROUGE-1 match evaluator

```python
from google.adk.evaluation.response_evaluator import ResponseEvaluator
from google.adk.evaluation.eval_case import Invocation
from google.genai import types

evaluator = ResponseEvaluator(
    threshold=0.5,
    metric_name="response_match_score",
)

actual = [
    Invocation(
        user_content=types.Content(
            role="user", parts=[types.Part(text="What is the capital of France?")]
        ),
        final_response=types.Content(
            role="model", parts=[types.Part(text="The capital of France is Paris.")]
        ),
    )
]
expected = [
    Invocation(
        user_content=types.Content(
            role="user", parts=[types.Part(text="What is the capital of France?")]
        ),
        final_response=types.Content(
            role="model", parts=[types.Part(text="Paris is the capital of France.")]
        ),
    )
]

result = evaluator.evaluate_invocations(actual, expected)
print(result.overall_eval_status)  # PASSED or FAILED
print(result.overall_score)        # ROUGE-1 F-measure
```

### Example 2 — coherence evaluator using `EvalMetric`

```python
from google.adk.evaluation.response_evaluator import ResponseEvaluator
from google.adk.evaluation.eval_metrics import EvalMetric, PrebuiltMetrics

evaluator = ResponseEvaluator(
    eval_metric=EvalMetric(
        metric_name=PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value,
        threshold=3.5,  # Coherence ≥ 3.5 / 5 → PASSED
    )
)
# evaluate_invocations() calls Vertex AI for coherence scoring
```

### Example 3 — running `ResponseEvaluator` inside `AgentEvaluator`

```python
import asyncio
from google.adk.evaluation.agent_evaluator import AgentEvaluator

async def evaluate():
    # evaluate() reads metrics from evals/test_config.json and asserts
    # failures internally; it returns None and prints a results table.
    await AgentEvaluator.evaluate(
        agent_module="my_agent",
        eval_dataset_file_path_or_dir="evals/",
        num_runs=1,
        print_detailed_results=True,
    )

asyncio.run(evaluate())
```

---

## 8 · `BaseSessionService` + `GetSessionConfig` + `ListSessionsResponse` — session service ABC and query types

**Source:** `google/adk/sessions/base_session_service.py`

`BaseSessionService` is the ABC all session backends implement:
`InMemorySessionService`, `SqliteSessionService`, `VertexAiSessionService`,
`FirestoreSessionService`, and `PerAgentDatabaseSessionService`. `GetSessionConfig`
controls event filtering in `get_session`; `ListSessionsResponse` is the
paginated sessions container.

### Method surface (verified `base_session_service.py`)

| Method | Returns | Description |
|--------|---------|-------------|
| `create_session(...)` | `Session` | Creates a new session, optionally with initial state and a client-provided ID |
| `get_session(...)` | `Optional[Session]` | Loads by ID; `config` limits returned events |
| `list_sessions(...)` | `ListSessionsResponse` | All sessions for a user (events/state stripped) |
| `delete_session(...)` | `None` | Permanently removes a session |
| `get_user_state(...)` | `dict` | User-scoped state shared across all sessions |
| `append_event(...)` | `Event` | Appends an event and updates session state |

### `GetSessionConfig` fields (verified `base_session_service.py`)

```python
class GetSessionConfig(BaseModel):
    num_recent_events: Optional[int] = None
    # None → all events; 0 → no events; N → last N events
    after_timestamp: Optional[float] = None
    # None → all events; float → events with timestamp >= value
```

### Example 1 — implementing a custom in-memory `BaseSessionService`

```python
from __future__ import annotations

import uuid
from typing import Any, Optional

from google.adk.events.event import Event
from google.adk.sessions.base_session_service import (
    BaseSessionService, GetSessionConfig, ListSessionsResponse,
)
from google.adk.sessions.session import Session


class SimpleSessionService(BaseSessionService):
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    async def create_session(
        self, *, app_name, user_id, state=None, session_id=None
    ) -> Session:
        sid = session_id or str(uuid.uuid4())
        session = Session(
            id=sid,
            app_name=app_name,
            user_id=user_id,
            state=state or {},
        )
        self._sessions[sid] = session
        return session

    async def get_session(
        self, *, app_name, user_id, session_id,
        config: Optional[GetSessionConfig] = None
    ) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if session is None or session.app_name != app_name or session.user_id != user_id:
            return None
        if config is None:
            return session
        events = session.events
        if config.after_timestamp is not None:
            events = [e for e in events if e.timestamp >= config.after_timestamp]
        if config.num_recent_events == 0:
            events = []
        elif config.num_recent_events is not None:
            events = events[-config.num_recent_events:]
        import copy
        filtered = copy.copy(session)
        filtered.events = events
        return filtered

    async def list_sessions(self, *, app_name, user_id=None) -> ListSessionsResponse:
        stripped = [
            Session(id=s.id, app_name=s.app_name, user_id=s.user_id)
            for s in self._sessions.values()
            if s.app_name == app_name and (user_id is None or s.user_id == user_id)
        ]
        return ListSessionsResponse(sessions=stripped)

    async def delete_session(self, *, app_name, user_id, session_id) -> None:
        self._sessions.pop(session_id, None)

    async def get_user_state(self, *, app_name, user_id) -> dict[str, Any]:
        return {}

    async def append_event(self, session: Session, event: Event) -> Event:
        # get_session may return a filtered copy; always persist to the stored
        # session so events survive across turns.
        stored = self._sessions.get(session.id, session)
        return await super().append_event(stored, event)
```

### Example 2 — `GetSessionConfig` to limit events returned

```python
import asyncio
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.in_memory_session_service import InMemorySessionService

async def demo():
    svc = InMemorySessionService()
    session = await svc.create_session(app_name="app", user_id="u1")

    # Load session without any events (e.g. for state-only inspection)
    lean = await svc.get_session(
        app_name="app",
        user_id="u1",
        session_id=session.id,
        config=GetSessionConfig(num_recent_events=0),
    )
    print("Events in lean view:", len(lean.events))  # 0

    # Load only the 5 most recent events
    recent = await svc.get_session(
        app_name="app",
        user_id="u1",
        session_id=session.id,
        config=GetSessionConfig(num_recent_events=5),
    )
    print("Events in recent view:", len(recent.events))

asyncio.run(demo())
```

### Example 3 — `ListSessionsResponse` with pagination hint

```python
import asyncio
from google.adk.sessions.in_memory_session_service import InMemorySessionService

async def list_all():
    svc = InMemorySessionService()
    for i in range(5):
        await svc.create_session(app_name="app", user_id="u1")

    response = await svc.list_sessions(app_name="app", user_id="u1")
    # Events and state are NOT populated in list results (per ABC contract)
    print(f"Found {len(response.sessions)} sessions")
    for s in response.sessions:
        print(f"  id={s.id}  events=stripped")

asyncio.run(list_all())
```

---

## 9 · `ArtifactVersion` + `BaseArtifactService` — artifact metadata and storage ABC

**Source:** `google/adk/artifacts/base_artifact_service.py`

`ArtifactVersion` is a Pydantic model describing one immutable version of a
stored artifact. `BaseArtifactService` is the abstract base class all artifact
backends implement: `InMemoryArtifactService`, `FileArtifactService`, and
`GcsArtifactService`.

### `ArtifactVersion` fields (verified `base_artifact_service.py`)

```python
class ArtifactVersion(BaseModel):
    # camelCase JSON aliases (alias_generator=to_camel)
    version: int                    # monotonically increasing, starts at 0
    canonical_uri: str              # backend URI (e.g. gs:// or file://)
    custom_metadata: dict[str, Any] # user-supplied key/value pairs
    create_time: float              # Unix timestamp (platform_time.get_time())
    mime_type: Optional[str]        # MIME type for binary payloads
```

### `BaseArtifactService` method surface (verified `base_artifact_service.py`)

| Method | Returns | Description |
|--------|---------|-------------|
| `save_artifact(...)` | `int` (version) | Persist a `types.Part`; auto-increments version |
| `load_artifact(...)` | `Optional[types.Part]` | Load by version or latest |
| `list_artifact_keys(...)` | `list[str]` | All filenames in a session/user scope |
| `delete_artifact(...)` | `None` | Delete all versions of a filename |
| `list_versions(...)` | `list[int]` | All version numbers for a filename |
| `list_artifact_versions(...)` | `list[ArtifactVersion]` | Versions with metadata |
| `get_artifact_version(...)` | `Optional[ArtifactVersion]` | Metadata for one version |

**Scope rule:** `session_id=None` → user-scoped artifacts (shared across all
sessions); `session_id="<id>"` → session-scoped artifacts.

### Example 1 — minimal `BaseArtifactService` implementation

```python
import asyncio
import time
from collections import defaultdict
from typing import Optional

from google.genai import types
from google.adk.artifacts.base_artifact_service import (
    ArtifactVersion, BaseArtifactService,
)


class DictArtifactService(BaseArtifactService):
    """In-memory artifact service for testing."""

    def __init__(self):
        self._store: dict = defaultdict(list)

    def _key(self, app_name, user_id, session_id, filename):
        # filenames starting with "user:" are user-scoped regardless of session_id
        scope = "__user__" if filename.startswith("user:") else (session_id or "__user__")
        return (app_name, user_id, scope, filename)

    async def save_artifact(
        self, *, app_name, user_id, filename,
        artifact: types.Part,
        session_id=None, custom_metadata=None,
    ) -> int:
        k = self._key(app_name, user_id, session_id, filename)
        version = len(self._store[k])
        # Capture mime_type from the artifact at write time (stable per version).
        mime_type = None
        if artifact.inline_data is not None:
            mime_type = artifact.inline_data.mime_type
        elif artifact.text is not None:
            mime_type = "text/plain"
        av = ArtifactVersion(
            version=version,
            canonical_uri=f"mem://{filename}/{version}",
            custom_metadata=custom_metadata or {},
            create_time=time.time(),
            mime_type=mime_type,
        )
        self._store[k].append((artifact, av))
        return version

    async def load_artifact(
        self, *, app_name, user_id, filename, session_id=None, version=None
    ) -> Optional[types.Part]:
        k = self._key(app_name, user_id, session_id, filename)
        versions = self._store.get(k, [])
        if not versions:
            return None
        idx = version if version is not None else len(versions) - 1
        entry = versions[idx] if 0 <= idx < len(versions) else None
        return entry[0] if entry is not None else None

    async def list_artifact_keys(self, *, app_name, user_id, session_id=None):
        # Always include user-scoped artifacts; also session-scoped when session_id given.
        user_prefix = (app_name, user_id, "__user__")
        keys = {k[3] for k in self._store if k[:3] == user_prefix}
        if session_id is not None:
            session_prefix = (app_name, user_id, session_id)
            keys |= {k[3] for k in self._store if k[:3] == session_prefix}
        return sorted(keys)

    async def delete_artifact(self, *, app_name, user_id, filename, session_id=None):
        self._store.pop(self._key(app_name, user_id, session_id, filename), None)

    async def list_versions(self, *, app_name, user_id, filename, session_id=None):
        k = self._key(app_name, user_id, session_id, filename)
        return list(range(len(self._store.get(k, []))))

    async def list_artifact_versions(self, *, app_name, user_id, filename, session_id=None):
        k = self._key(app_name, user_id, session_id, filename)
        # Return the ArtifactVersion captured at save time (stable create_time/mime_type).
        return [av for _, av in self._store.get(k, [])]

    async def get_artifact_version(
        self, *, app_name, user_id, filename, session_id=None, version=None
    ) -> Optional[ArtifactVersion]:
        k = self._key(app_name, user_id, session_id, filename)
        versions = self._store.get(k, [])
        idx = version if version is not None else len(versions) - 1
        if 0 <= idx < len(versions):
            _, av = versions[idx]
            return av
        return None


async def demo():
    svc = DictArtifactService()
    v0 = await svc.save_artifact(
        app_name="app", user_id="u1", filename="report.txt",
        artifact=types.Part(text="v1 content"),
    )
    v1 = await svc.save_artifact(
        app_name="app", user_id="u1", filename="report.txt",
        artifact=types.Part(text="v2 content"),
    )
    latest = await svc.load_artifact(app_name="app", user_id="u1", filename="report.txt")
    print(f"Saved {v0}, {v1}; latest = {latest.text!r}")
    avs = await svc.list_artifact_versions(app_name="app", user_id="u1", filename="report.txt")
    print([av.version for av in avs])  # [0, 1]

asyncio.run(demo())
```

### Example 2 — `ArtifactVersion` with `custom_metadata` and camelCase JSON

```python
from google.adk.artifacts.base_artifact_service import ArtifactVersion
import time

av = ArtifactVersion(
    version=0,
    canonical_uri="gs://my-bucket/app/u1/report.pdf",
    custom_metadata={
        "generated_by": "report_agent",
        "page_count": 12,
        "tags": ["quarterly", "finance"],
    },
    create_time=time.time(),
    mime_type="application/pdf",
)

print(av.version)                        # 0
print(av.custom_metadata["page_count"])  # 12

# camelCase JSON (alias_generator=to_camel)
json_str = av.model_dump_json(by_alias=True)
restored = ArtifactVersion.model_validate_json(json_str)
assert restored.canonical_uri == av.canonical_uri
print("Round-trip OK")
```

### Example 3 — wiring `InMemoryArtifactService` into `InMemoryRunner`

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

agent = LlmAgent(
    name="archivist",
    model="gemini-2.5-flash",
    instruction="Save reports when asked.",
)

async def main():
    artifact_svc = InMemoryArtifactService()
    runner = Runner(
        agent=agent,
        artifact_service=artifact_svc,
        session_service=InMemorySessionService(),
        app_name="archivist_app",
    )
    session = await runner.session_service.create_session(
        app_name="archivist_app", user_id="u1"
    )
    # After tool calls, artifacts accumulate in artifact_svc.
    keys = await artifact_svc.list_artifact_keys(
        app_name="archivist_app", user_id="u1", session_id=session.id
    )
    print("Stored artifact keys:", keys)
    await runner.close()

asyncio.run(main())
```

---

## 10 · `EvalSet` — evaluation case container

**Source:** `google/adk/evaluation/eval_set.py`

`EvalSet` groups a list of `EvalCase` instances under a shared identifier. It
is the unit of persistence used by `EvalSetsManager` and accepted by
`AgentEvaluator`. Backends include `InMemoryEvalSetsManager` and
`GcsEvalSetsManager`.

### `EvalSet` fields (verified `eval_set.py`)

```python
class EvalSet(BaseModel):
    eval_set_id: str              # unique identifier; used as filename in some backends
    name: Optional[str] = None
    description: Optional[str] = None
    eval_cases: list[EvalCase]    # one EvalCase per test scenario
    creation_timestamp: float = 0.0
```

### Example 1 — building an `EvalSet` programmatically

```python
import time
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.genai import types


def make_case(eval_id: str, question: str, answer: str) -> EvalCase:
    return EvalCase(
        eval_id=eval_id,
        conversation=[
            Invocation(
                user_content=types.Content(
                    role="user", parts=[types.Part(text=question)]
                ),
                final_response=types.Content(
                    role="model", parts=[types.Part(text=answer)]
                ),
            )
        ],
    )


eval_set = EvalSet(
    eval_set_id="math_basics_v1",
    name="Math Basics",
    description="Simple arithmetic questions for the calculator agent.",
    eval_cases=[
        make_case("add_001", "What is 2+2?", "4"),
        make_case("mul_001", "What is 3×7?", "21"),
        make_case("div_001", "What is 10÷2?", "5"),
    ],
    creation_timestamp=time.time(),
)
print(f"{eval_set.eval_set_id}: {len(eval_set.eval_cases)} cases")
```

### Example 2 — JSON round-trip for `EvalSet`

```python
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.genai import types

es = EvalSet(
    eval_set_id="round_trip_test",
    eval_cases=[
        EvalCase(
            eval_id="greet_001",
            conversation=[
                Invocation(
                    user_content=types.Content(
                        role="user", parts=[types.Part(text="Hi")]
                    ),
                    final_response=types.Content(
                        role="model", parts=[types.Part(text="Hello!")]
                    ),
                )
            ],
        )
    ],
)

raw = es.model_dump_json(indent=2)
restored = EvalSet.model_validate_json(raw)
assert restored.eval_set_id == "round_trip_test"
assert len(restored.eval_cases) == 1
print("Round-trip OK")
```

### Example 3 — persisting and retrieving an `EvalSet` via `InMemoryEvalSetsManager`

```python
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.genai import types


def main():
    manager = InMemoryEvalSetsManager()

    # 1. Create the empty set
    manager.create_eval_set(app_name="demo_app", eval_set_id="demo_set")

    # 2. Add cases individually
    case = EvalCase(
        eval_id="ping_001",
        conversation=[
            Invocation(
                user_content=types.Content(
                    role="user", parts=[types.Part(text="ping")]
                ),
                final_response=types.Content(
                    role="model", parts=[types.Part(text="pong")]
                ),
            )
        ],
    )
    manager.add_eval_case(
        app_name="demo_app", eval_set_id="demo_set", eval_case=case
    )

    # 3. Retrieve and inspect
    retrieved: EvalSet = manager.get_eval_set(
        app_name="demo_app", eval_set_id="demo_set"
    )
    print(f"Retrieved {len(retrieved.eval_cases)} case(s)")
    print(f"First case: {retrieved.eval_cases[0].eval_id}")

main()
```

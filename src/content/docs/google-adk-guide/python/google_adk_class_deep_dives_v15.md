---
title: "Class deep dives — volume 15 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.2.0 classes: LlmRequest (request builder; append_instructions/append_tools/set_output_schema; cache_config wiring), LlmResponse (streaming contract; partial/turn_complete; get_function_calls; error fields; usage_metadata), BaseLlm (custom model backend contract; generate_content_async streaming semantics; connect for live; supported_models regex), LLMRegistry (lazy-loaded model resolution; prefix syntax; register custom backends; lru_cache), BaseLlmFlow/AutoFlow/SingleFlow (LLM orchestration pipeline; request/response processor chains; agent-transfer wiring), BaseLlmRequestProcessor/BaseLlmResponseProcessor (pipeline stage protocol; custom request transforms; response post-processing), Sampler/AgentOptimizer (abstract optimization protocol; sample_and_score; train/validation split), LocalEvalSampler/LocalEvalSamplerConfig (ADK eval service as optimization sampler; full optimization loop wiring), SamplingResult/AgentWithScores/OptimizerResult/UnstructuredSamplingResult (optimization data model; Pareto-front results; per-example score maps), ApiRegistry (GCP Cloud API Registry integration; paginated MCP server discovery; get_toolset with tool_filter)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 15"
  order: 80
---

Source-verified against **google-adk==2.2.0** (installed from PyPI, June 2026). Every field name, signature, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `LlmRequest` | `google.adk.models.llm_request` | Stable |
| 2 | `LlmResponse` | `google.adk.models.llm_response` | Stable |
| 3 | `BaseLlm` | `google.adk.models.base_llm` | Stable |
| 4 | `LLMRegistry` | `google.adk.models.registry` | Stable |
| 5 | `BaseLlmFlow` + `AutoFlow` + `SingleFlow` | `google.adk.flows.llm_flows` | Stable |
| 6 | `BaseLlmRequestProcessor` + `BaseLlmResponseProcessor` | `google.adk.flows.llm_flows._base_llm_processor` | Stable |
| 7 | `Sampler` + `AgentOptimizer` | `google.adk.optimization.sampler`, `.agent_optimizer` | Experimental |
| 8 | `LocalEvalSampler` + `LocalEvalSamplerConfig` | `google.adk.optimization.local_eval_sampler` | Experimental |
| 9 | `SamplingResult` + `AgentWithScores` + `OptimizerResult` + `UnstructuredSamplingResult` | `google.adk.optimization.data_types` | Experimental |
| 10 | `ApiRegistry` | `google.adk.integrations.api_registry.api_registry` | Experimental |

---

## 1 · `LlmRequest`

**Source:** `google.adk.models.llm_request`

`LlmRequest` is the internal envelope passed through every stage of the ADK's LLM pipeline. It carries the conversation history (`contents`), model configuration (`config`), registered tools (`tools_dict`), context-cache settings, and optional live-session configuration. Understanding its mutation API is essential whenever you write a custom `BaseLlmRequestProcessor`, a planner, or any code that needs to inspect or modify what gets sent to the model.

### Fields (source-verified)

```python
from google.adk.models.llm_request import LlmRequest
from google.genai import types

LlmRequest(
    model: str | None = None,                          # model name, e.g. "gemini-2.5-flash"
    contents: list[types.Content] = [],                # conversation history
    config: types.GenerateContentConfig = ...,         # system_instruction, tools, temperature, …
    live_connect_config: types.LiveConnectConfig = ...,# live/bidi session settings
    tools_dict: dict[str, BaseTool] = {},              # name→tool lookup for function dispatch
    cache_config: ContextCacheConfig | None = None,    # opt-in context caching
    cache_metadata: CacheMetadata | None = None,       # cache state from previous turn
    cacheable_contents_token_count: int | None = None, # token count for cache-size gating
    previous_interaction_id: str | None = None,        # stateful Interactions API chaining
)
```

### `append_instructions` — add to system prompt

```python
# Simple string list — concatenates with "\n\n"
req = LlmRequest(model="gemini-2.5-flash")
req.append_instructions(["You are a helpful assistant.", "Always reply in JSON."])
print(req.config.system_instruction)
# "You are a helpful assistant.\n\nAlways reply in JSON."

# Idempotent accumulation across multiple calls
req.append_instructions(["Language: English only."])
print(req.config.system_instruction)
# "You are a helpful assistant.\n\nAlways reply in JSON.\n\nLanguage: English only."
```

### `append_instructions` with `types.Content` (multimodal system prompt)

```python
import base64
from google.genai import types

logo_bytes = open("logo.png", "rb").read()

instruction_with_image = types.Content(
    parts=[
        types.Part(text="You are a brand voice assistant. Match this brand style:"),
        types.Part(inline_data=types.Blob(
            mime_type="image/png",
            data=base64.b64encode(logo_bytes).decode(),
            display_name="brand-logo",
        )),
        types.Part(text="Always produce output consistent with the visual identity above."),
    ]
)

req = LlmRequest(model="gemini-2.5-flash")
user_contents = req.append_instructions(instruction_with_image)
# Text parts → system_instruction string
# Blob parts → returned as user-role Content objects to be prepended
# user_contents contains types.Content(role="user", parts=[reference, blob])
```

### `append_tools` — register tools and add declarations

```python
from google.adk.tools import FunctionTool

def lookup_order(order_id: str) -> dict:
    """Returns order details for the given order ID."""
    return {"status": "shipped", "eta": "2 days"}

order_tool = FunctionTool(func=lookup_order)

req = LlmRequest(model="gemini-2.5-flash")
req.append_tools([order_tool])

# Declaration added to config.tools for model schema discovery
assert req.config.tools is not None
# Tool registered for function-call dispatch
assert "lookup_order" in req.tools_dict
```

### `set_output_schema` — constrained JSON output

```python
from pydantic import BaseModel
from typing import List

class ProductRecommendation(BaseModel):
    product_name: str
    reason: str
    confidence: float

class RecommendationList(BaseModel):
    recommendations: List[ProductRecommendation]

req = LlmRequest(model="gemini-2.5-flash")
req.set_output_schema(RecommendationList)

# Sets response_mime_type + response_schema on config
assert req.config.response_mime_type == "application/json"
# Model will now return structured JSON matching RecommendationList
```

### Building a complete request for a custom tool

```python
from google.adk.models.llm_request import LlmRequest
from google.adk.tools import FunctionTool
from google.genai import types

def search_documents(query: str, max_results: int = 5) -> list[dict]:
    """Searches the knowledge base for relevant documents."""
    return [{"title": "doc1", "snippet": "..."}]

req = LlmRequest(model="gemini-2.5-pro")
req.append_instructions(["You are a research assistant with access to a document library."])
req.append_tools([FunctionTool(func=search_documents)])
req.set_output_schema({"type": "object", "properties": {"answer": {"type": "string"}, "sources": {"type": "array"}}})

req.contents.append(
    types.Content(
        role="user",
        parts=[types.Part(text="What are the best practices for rate limiting?")],
    )
)
```

---

## 2 · `LlmResponse`

**Source:** `google.adk.models.llm_response`

`LlmResponse` is the object yielded by every `BaseLlm.generate_content_async` call. In **streaming mode** the generator yields multiple `LlmResponse` objects — intermediate ones have `partial=True`, and the final one has `partial=False` and contains the fully-aggregated content. Non-streaming mode yields exactly one response.

### Fields (source-verified)

```python
LlmResponse(
    model_version: str | None = None,
    content: types.Content | None = None,          # model output (text, function calls, …)
    grounding_metadata: types.GroundingMetadata | None = None,
    partial: bool | None = None,                   # True for streaming chunks
    turn_complete: bool | None = None,             # live/bidi only
    turn_complete_reason: types.TurnCompleteReason | None = None,
    finish_reason: types.FinishReason | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    interrupted: bool | None = None,               # live bidi interruption
    custom_metadata: dict | None = None,
    usage_metadata: types.GenerateContentResponseUsageMetadata | None = None,
    live_session_resumption_update: ... | None = None,
    live_session_id: str | None = None,
    go_away: types.LiveServerGoAway | None = None,
    input_transcription: types.Transcription | None = None,
    output_transcription: types.Transcription | None = None,
    avg_logprobs: float | None = None,
    logprobs_result: types.LogprobsResult | None = None,
    cache_metadata: CacheMetadata | None = None,   # set when context cache was used
    citation_metadata: types.CitationMetadata | None = None,
    interaction_id: str | None = None,
)
```

### Reading streaming chunks

```python
from google.adk.models.llm_response import LlmResponse

# Simulated streaming output from BaseLlm.generate_content_async
async def consume_streaming(llm, request):
    tokens = []
    final_response = None

    async for response in llm.generate_content_async(request, stream=True):
        if response.partial:
            # Intermediate chunk — accumulate text
            if response.content and response.content.parts:
                for part in response.content.parts:
                    if part.text and not getattr(part, "thought", False):
                        tokens.append(part.text)
        else:
            # Final chunk — fully-aggregated result
            final_response = response

    return "".join(tokens), final_response
```

### Extracting function calls

```python
async def handle_tool_turn(llm, request):
    response = None
    async for r in llm.generate_content_async(request, stream=False):
        response = r

    # Check for function calls
    function_calls = response.get_function_calls()
    if function_calls:
        for fc in function_calls:
            print(f"Tool requested: {fc.name}, args: {fc.args}")
    elif response.error_code:
        print(f"Model error: {response.error_code} — {response.error_message}")
    else:
        print(f"Text response: {response.content.parts[0].text if response.content else 'empty'}")
```

### Checking usage and cache metadata

```python
async def audit_response(llm, request):
    async for response in llm.generate_content_async(request, stream=False):
        if response.usage_metadata:
            um = response.usage_metadata
            print(f"Input tokens:  {um.prompt_token_count}")
            print(f"Output tokens: {um.candidates_token_count}")
            print(f"Total:         {um.total_token_count}")
            if um.cached_content_token_count:
                saved = um.prompt_token_count - um.cached_content_token_count
                print(f"Cache saved:   {saved} tokens")

        if response.cache_metadata:
            print(f"Cache name:    {response.cache_metadata.cache_name}")
            print(f"Invocations:   {response.cache_metadata.invocations_used}")
```

### `LlmResponse.create` factory — from raw Gemini response

```python
from google.adk.models.llm_response import LlmResponse
from google.genai import types

# ADK uses this internally to wrap the Gemini SDK response
raw = await client.aio.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello",
)
response: LlmResponse = LlmResponse.create(raw)
# response.content contains the first candidate's content
# response.error_code is set if finish_reason was not STOP
```

### Handling grounded responses

```python
async def grounded_search(llm, request):
    async for response in llm.generate_content_async(request, stream=False):
        if response.grounding_metadata:
            gm = response.grounding_metadata
            sources = gm.grounding_chunks or []
            for chunk in sources:
                if chunk.web:
                    print(f"Source: {chunk.web.uri} — {chunk.web.title}")
        if response.citation_metadata:
            for citation in (response.citation_metadata.citation_sources or []):
                print(f"Citation: {citation.uri} (start={citation.start_index})")
```

---

## 3 · `BaseLlm`

**Source:** `google.adk.models.base_llm`

`BaseLlm` is the abstract contract every LLM backend must implement to integrate with ADK. Implementing it correctly gives you access to the full ADK ecosystem: planners, tools, compaction, eval, caching, and streaming — without any changes to agent code.

### Class definition (source-verified)

```python
from abc import abstractmethod
from typing import AsyncGenerator
from google.genai import types
from pydantic import BaseModel, ConfigDict
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.base_llm_connection import BaseLlmConnection

class BaseLlm(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str  # required

    @classmethod
    def supported_models(cls) -> list[str]:
        """Regex patterns for LLMRegistry auto-registration."""
        return []

    @abstractmethod
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]: ...

    def connect(self, llm_request: LlmRequest) -> BaseLlmConnection:
        """Override for live/bidi sessions (e.g. Gemini Live API)."""
        raise NotImplementedError(f"Live connection not supported for {self.model}.")
```

### Implementing a custom `BaseLlm` backend

```python
from typing import AsyncGenerator
import httpx
from google.genai import types
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry

class MyOpenAICompatibleLlm(BaseLlm):
    """Routes requests to any OpenAI-compatible endpoint."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str

    @classmethod
    def supported_models(cls) -> list[str]:
        # Regex: matches "myapi/..." — prefix routing
        return [r"myapi/.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        # Convert ADK LlmRequest to OpenAI-style messages
        messages = []
        if llm_request.config and llm_request.config.system_instruction:
            messages.append({"role": "system", "content": llm_request.config.system_instruction})
        for content in llm_request.contents:
            for part in content.parts:
                if part.text:
                    messages.append({"role": content.role, "content": part.text})

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages},
            )
            data = resp.json()
            text = data["choices"][0]["message"]["content"]

        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=text)],
            ),
            partial=False,
        )

# Register so LLMRegistry.resolve("myapi/gpt-4o") returns this class
LLMRegistry.register(MyOpenAICompatibleLlm)
```

### Using `_maybe_append_user_content`

```python
# BaseLlm includes this helper to satisfy the requirement that
# the conversation always ends with a user turn before the model responds.

class MyLlm(BaseLlm):
    async def generate_content_async(self, llm_request, stream=False):
        # Ensure model has something to respond to
        self._maybe_append_user_content(llm_request)
        # ... send to backend
        yield LlmResponse(...)
```

---

## 4 · `LLMRegistry`

**Source:** `google.adk.models.registry`

`LLMRegistry` is the global factory that maps model name strings to `BaseLlm` subclasses. It supports **prefix routing** (`openai:gpt-4o`), **regex pattern matching** (e.g. `gemini-.*`), and **lazy imports** so optional dependencies (litellm, anthropic) are only loaded when actually needed.

### `new_llm` — create an LLM instance by name

```python
from google.adk.models.registry import LLMRegistry

# Resolves via regex: "gemini-.*" → Gemini class
gemini = LLMRegistry.new_llm("gemini-2.5-flash")

# Prefix routing: "openai:" prefix → LiteLlm, strips prefix before passing model name
openai_llm = LLMRegistry.new_llm("openai/gpt-4o")  # requires google-adk[extensions]

# Explicit prefix override: "litellm:ollama/llama3.2"
ollama_llm = LLMRegistry.new_llm("litellm:ollama/llama3.2")
```

### `resolve` — get the class without instantiating

```python
# resolve is lru_cache(maxsize=32) — fast on repeated calls
from google.adk.models.registry import LLMRegistry

cls = LLMRegistry.resolve("gemini-2.5-pro")
print(cls.__name__)  # "Gemini"

# Use to inspect supported models
instance = cls(model="gemini-2.5-pro")
```

### `register` — add a custom backend

```python
from google.adk.models.registry import LLMRegistry
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from typing import AsyncGenerator
from google.genai import types

class MockLlm(BaseLlm):
    """Deterministic test backend."""

    response_text: str = "mock response"

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"mock-.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=self.response_text)],
            ),
            partial=False,
        )

LLMRegistry.register(MockLlm)

# Now usable anywhere an Agent accepts a model string
from google.adk.agents import LlmAgent
agent = LlmAgent(name="test_agent", model="mock-v1", instruction="Say hello.")
```

### Prefix routing internals

```python
# Prefix format: "provider:model_name"
# LLMRegistry._parse_model("openai:gpt-4o") → ("openai", "gpt-4o")
# LLMRegistry._match_prefix("openai", "LiteLlm") → True  (strips "llm" suffix)

# The resolved class receives just the model part:
#   LiteLlm(model="gpt-4o")   (not LiteLlm(model="openai:gpt-4o"))

# Lazy-registered entries are (module_path, class_name) tuples that are
# imported on first resolve and cached back into the dict:
LLMRegistry._register_lazy(
    [r"myvendor/.*"],
    "mypackage.mymodule",
    "MyVendorLlm",
)
```

### Providing helpful error messages for unknown models

```python
# LLMRegistry.resolve raises ValueError with provider-specific hints:
# - "claude-*" → "Install anthropic>=0.43.0 or google-adk[extensions]"
# - "provider/model" → "Install litellm>=1.75.5 or google-adk[extensions]"

try:
    LLMRegistry.new_llm("claude-opus-4-8")
except ValueError as e:
    print(e)
    # "Model claude-opus-4-8 not found.
    #  Claude models require the anthropic package.
    #  Install it with: pip install google-adk[extensions]"
```

---

## 5 · `BaseLlmFlow` + `AutoFlow` + `SingleFlow`

**Source:** `google.adk.flows.llm_flows`

`BaseLlmFlow` is the abstract orchestrator that drives a single LLM turn. `SingleFlow` builds the standard processor pipeline (system instructions, compaction, context caching, code execution, NL planning, output schema). `AutoFlow` extends `SingleFlow` with the **agent-transfer** processor, enabling sub-agent routing.

### Processor pipeline in `SingleFlow` (source-verified)

```
Request processors (in order):
  1. basic             — sets model, config defaults
  2. auth_preprocessor — injects auth credentials
  3. request_confirmation — tool confirmation gates
  4. instructions      — appends agent system instruction
  5. identity          — adds agent identity metadata
  6. compaction        — context window compaction
  7. contents          — attaches conversation history
  8. context_cache     — wires GeminiContextCacheManager
  9. interactions      — stateful Interactions API chain
 10. _nl_planning      — PlanReActPlanner instruction injection
 11. _code_execution   — extracts CSV data files
 12. _output_schema    — output schema + SetModelResponseTool

Response processors (in order):
  1. _nl_planning      — marks planning thoughts
  2. _code_execution   — executes code blocks, injects results
```

### `AutoFlow` adds agent transfer

```python
# AutoFlow extends SingleFlow with one extra request processor:
from google.adk.flows.llm_flows.auto_flow import AutoFlow
from google.adk.flows.llm_flows.single_flow import SingleFlow

# AutoFlow processor list = SingleFlow's list + [agent_transfer.request_processor]
# agent_transfer injects sibling/parent agent names into the system prompt
# and builds the transfer_to_agent tool declaration automatically.

# LlmAgent uses AutoFlow by default when sub-agents or a parent are present.
```

### Extending `SingleFlow` with a custom processor

```python
from google.adk.flows.llm_flows.single_flow import SingleFlow
from google.adk.flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.events.event import Event
from typing import AsyncGenerator

class PiiRedactionProcessor(BaseLlmRequestProcessor):
    """Redacts PII patterns from all user contents before they reach the model."""

    async def run_async(
        self, invocation_context: InvocationContext, llm_request: LlmRequest
    ) -> AsyncGenerator[Event, None]:
        import re
        PII_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")  # SSN pattern

        for content in llm_request.contents:
            if content.role == "user":
                for part in content.parts:
                    if part.text:
                        part.text = PII_PATTERN.sub("[REDACTED-SSN]", part.text)
        # Processors must be async generators — yield nothing if no events to emit
        return
        yield  # required for AsyncGenerator type

pii_processor = PiiRedactionProcessor()

class SecureFlow(SingleFlow):
    def __init__(self):
        super().__init__()
        # Insert PII redaction before content history is attached
        self.request_processors.insert(3, pii_processor)
```

### Wiring a custom flow to an agent

```python
from google.adk.agents import LlmAgent

class MyAgent(LlmAgent):
    def _get_flow(self):
        return SecureFlow()  # override the default AutoFlow

agent = MyAgent(
    name="secure_support_agent",
    model="gemini-2.5-flash",
    instruction="You are a support agent. Handle customer queries.",
)
```

---

## 6 · `BaseLlmRequestProcessor` + `BaseLlmResponseProcessor`

**Source:** `google.adk.flows.llm_flows._base_llm_processor`

The processor protocol is the standard extension point for injecting logic into the LLM call pipeline. Both abstract classes expose a single `run_async` method that is an async generator: processors can **yield `Event` objects** (e.g., to emit intermediate auth-request events) or simply return without yielding.

### `BaseLlmRequestProcessor` — mutate the outgoing request

```python
from typing import AsyncGenerator
from google.adk.flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.events.event import Event

class TokenBudgetRequestProcessor(BaseLlmRequestProcessor):
    """Enforces a maximum token budget by truncating old history."""

    max_contents: int = 20

    async def run_async(
        self, invocation_context: InvocationContext, llm_request: LlmRequest
    ) -> AsyncGenerator[Event, None]:
        if len(llm_request.contents) > self.max_contents:
            # Keep the most recent contents; discard the oldest
            dropped = len(llm_request.contents) - self.max_contents
            llm_request.contents = llm_request.contents[dropped:]
        return
        yield
```

### `BaseLlmResponseProcessor` — post-process the model output

```python
from google.adk.flows.llm_flows._base_llm_processor import BaseLlmResponseProcessor
from google.adk.models.llm_response import LlmResponse
import logging

logger = logging.getLogger(__name__)

class UsageAuditResponseProcessor(BaseLlmResponseProcessor):
    """Logs token usage for every non-streaming model response."""

    async def run_async(
        self, invocation_context: InvocationContext, llm_response: LlmResponse
    ) -> AsyncGenerator[Event, None]:
        if not llm_response.partial and llm_response.usage_metadata:
            um = llm_response.usage_metadata
            logger.info(
                "agent=%s tokens: in=%s out=%s total=%s cached=%s",
                invocation_context.agent.name,
                um.prompt_token_count,
                um.candidates_token_count,
                um.total_token_count,
                um.cached_content_token_count or 0,
            )
        return
        yield
```

### Emitting events from a processor

```python
from google.adk.events.event import Event
from google.genai import types

class ContentWarningProcessor(BaseLlmResponseProcessor):
    """Emits a warning event when the model output contains policy violations."""

    VIOLATION_KEYWORDS = {"harmful", "dangerous"}

    async def run_async(
        self, invocation_context: InvocationContext, llm_response: LlmResponse
    ) -> AsyncGenerator[Event, None]:
        if not llm_response.partial and llm_response.content:
            for part in llm_response.content.parts:
                if part.text:
                    text_lower = part.text.lower()
                    if any(kw in text_lower for kw in self.VIOLATION_KEYWORDS):
                        # Yield a synthetic model event flagging the violation
                        yield Event(
                            author=invocation_context.agent.name,
                            content=types.Content(
                                role="model",
                                parts=[types.Part(text="[Content policy violation flagged]")],
                            ),
                        )
                        return
```

### Full pipeline registration example

```python
from google.adk.flows.llm_flows.single_flow import SingleFlow

class AuditedFlow(SingleFlow):
    def __init__(self):
        super().__init__()
        self.request_processors.insert(0, TokenBudgetRequestProcessor())
        self.response_processors.append(UsageAuditResponseProcessor())
        self.response_processors.append(ContentWarningProcessor())
```

---

## 7 · `Sampler` + `AgentOptimizer`

**Source:** `google.adk.optimization.sampler`, `google.adk.optimization.agent_optimizer`

`Sampler` and `AgentOptimizer` are the two abstract base classes that define ADK's **prompt optimisation protocol**. `Sampler` provides the evaluation harness — scoring candidate agents against labelled examples. `AgentOptimizer` consumes a `Sampler` and implements an optimisation strategy (e.g., iterative LLM rewriting, GEPA).

### `Sampler` abstract interface

```python
from abc import ABC, abstractmethod
from typing import Literal, Optional, Generic
from google.adk.agents.llm_agent import Agent
from google.adk.optimization.data_types import SamplingResultT

class Sampler(ABC, Generic[SamplingResultT]):
    TRAIN_SET = "train"
    VALIDATION_SET = "validation"

    @abstractmethod
    def get_train_example_ids(self) -> list[str]: ...

    @abstractmethod
    def get_validation_example_ids(self) -> list[str]: ...

    @abstractmethod
    async def sample_and_score(
        self,
        candidate: Agent,
        example_set: Literal["train", "validation"] = "validation",
        batch: Optional[list[str]] = None,
        capture_full_eval_data: bool = False,
    ) -> SamplingResultT: ...
```

### Implementing a custom `Sampler`

```python
import random
from google.adk.optimization.sampler import Sampler
from google.adk.optimization.data_types import UnstructuredSamplingResult
from google.adk.agents.llm_agent import Agent

# Suppose you have a list of labelled test cases in memory
EXAMPLES = {
    "ex1": {"prompt": "Summarize this article: ...", "expected": "Short summary."},
    "ex2": {"prompt": "Translate to Spanish: hello", "expected": "hola"},
    "ex3": {"prompt": "What is 2+2?", "expected": "4"},
    "ex4": {"prompt": "Capital of France?", "expected": "Paris"},
    "ex5": {"prompt": "Who wrote Hamlet?", "expected": "Shakespeare"},
}

class InMemorySampler(Sampler[UnstructuredSamplingResult]):
    """Simple sampler that scores candidates against fixed in-memory examples."""

    def get_train_example_ids(self) -> list[str]:
        # Use 80% for training
        ids = list(EXAMPLES.keys())
        return ids[:4]

    def get_validation_example_ids(self) -> list[str]:
        ids = list(EXAMPLES.keys())
        return ids[4:]

    async def sample_and_score(
        self,
        candidate: Agent,
        example_set="validation",
        batch=None,
        capture_full_eval_data=False,
    ) -> UnstructuredSamplingResult:
        ids = batch or (
            self.get_train_example_ids() if example_set == "train"
            else self.get_validation_example_ids()
        )
        scores = {}
        for eid in ids:
            ex = EXAMPLES[eid]
            # Run candidate and compare output (simplified)
            output = await self._run_agent(candidate, ex["prompt"])
            scores[eid] = 1.0 if ex["expected"].lower() in output.lower() else 0.0
        return UnstructuredSamplingResult(scores=scores)

    async def _run_agent(self, agent: Agent, prompt: str) -> str:
        # In production: use ADK Runner to get the final text response
        return f"[agent output for: {prompt}]"
```

### `AgentOptimizer` abstract interface

```python
from abc import ABC, abstractmethod
from typing import Generic
from google.adk.agents.llm_agent import Agent
from google.adk.optimization.data_types import AgentWithScoresT, SamplingResultT, OptimizerResult
from google.adk.optimization.sampler import Sampler

class AgentOptimizer(ABC, Generic[SamplingResultT, AgentWithScoresT]):

    @abstractmethod
    async def optimize(
        self,
        initial_agent: Agent,
        sampler: Sampler[SamplingResultT],
    ) -> OptimizerResult[AgentWithScoresT]:
        """Run the optimization loop and return the best agent(s) with scores."""
        ...
```

### Implementing a custom `AgentOptimizer`

```python
from google.adk.optimization.agent_optimizer import AgentOptimizer
from google.adk.optimization.data_types import (
    AgentWithScores, OptimizerResult, UnstructuredSamplingResult
)

class RandomSearchOptimizer(
    AgentOptimizer[UnstructuredSamplingResult, AgentWithScores]
):
    """Randomly samples prompt variations and returns the best scoring one."""

    PROMPT_VARIANTS = [
        "You are a concise assistant. Answer briefly.",
        "You are a helpful assistant. Be thorough.",
        "You are an expert. Provide detailed explanations.",
    ]

    async def optimize(
        self, initial_agent, sampler
    ) -> OptimizerResult[AgentWithScores]:
        best_agent = initial_agent
        best_score = 0.0

        for variant_prompt in self.PROMPT_VARIANTS:
            candidate = initial_agent.clone(update={"instruction": variant_prompt})
            result = await sampler.sample_and_score(candidate, "train")
            avg = sum(result.scores.values()) / len(result.scores) if result.scores else 0.0
            if avg > best_score:
                best_score = avg
                best_agent = candidate

        # Final validation score
        val_result = await sampler.sample_and_score(best_agent, "validation")
        val_score = (
            sum(val_result.scores.values()) / len(val_result.scores)
            if val_result.scores else 0.0
        )

        return OptimizerResult(
            optimized_agents=[AgentWithScores(optimized_agent=best_agent, overall_score=val_score)]
        )
```

---

## 8 · `LocalEvalSampler` + `LocalEvalSamplerConfig`

**Source:** `google.adk.optimization.local_eval_sampler`

`LocalEvalSampler` is the **bridge between the ADK evaluation system and the optimizer**. It wraps `LocalEvalService` so that any `AgentOptimizer` (including `SimplePromptOptimizer` and `GEPARootAgentPromptOptimizer`) can score candidates using your existing eval sets — without any custom sampler code.

### `LocalEvalSamplerConfig` fields

```python
from google.adk.optimization.local_eval_sampler import LocalEvalSamplerConfig
from google.adk.evaluation.eval_config import EvalConfig

LocalEvalSamplerConfig(
    eval_config: EvalConfig,            # required — metrics, thresholds, judge model
    app_name: str,                      # required — matches your eval set app name
    train_eval_set: str,                # required — eval set ID used for training
    train_eval_case_ids: list[str] | None = None,   # subset of cases (default: all)
    validation_eval_set: str | None = None,          # separate val set (default: reuse train)
    validation_eval_case_ids: list[str] | None = None,
)
```

### Full optimization loop with `LocalEvalSampler`

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import EvalMetric, MetricName
from google.adk.evaluation.local_eval_sets_manager import LocalEvalSetsManager
from google.adk.optimization.local_eval_sampler import LocalEvalSampler, LocalEvalSamplerConfig
from google.adk.optimization.simple_prompt_optimizer import (
    SimplePromptOptimizer, SimplePromptOptimizerConfig
)

APP_NAME = "customer_support"
EVAL_SET_ID = "support_cases_v2"

async def optimize_agent():
    # 1. Load eval sets from your local .evalset.json files
    eval_sets_manager = LocalEvalSetsManager(eval_sets_path="./eval_sets/")

    # 2. Configure what metrics to use when scoring candidates
    eval_config = EvalConfig(
        eval_metrics=[
            EvalMetric(metric_name=MetricName.TOOL_CALL_QUALITY),
            EvalMetric(metric_name=MetricName.RESPONSE_QUALITY_RUBRIC),
        ]
    )

    # 3. Build the sampler
    sampler_config = LocalEvalSamplerConfig(
        eval_config=eval_config,
        app_name=APP_NAME,
        train_eval_set=EVAL_SET_ID,
        # Use 70% for training; rest for validation
        train_eval_case_ids=["case_001", "case_002", "case_003", "case_004", "case_005",
                             "case_006", "case_007"],
        validation_eval_set=EVAL_SET_ID,
        validation_eval_case_ids=["case_008", "case_009", "case_010"],
    )
    sampler = LocalEvalSampler(
        config=sampler_config,
        eval_sets_manager=eval_sets_manager,
    )

    # 4. Build the initial agent
    initial_agent = LlmAgent(
        name=APP_NAME,
        model="gemini-2.5-flash",
        instruction=(
            "You are a customer support agent. "
            "Help users resolve their issues with our product."
        ),
    )

    # 5. Run the optimizer
    optimizer = SimplePromptOptimizer(
        config=SimplePromptOptimizerConfig(
            optimizer_model="gemini-2.5-flash",
            num_iterations=5,
            batch_size=3,
        )
    )
    result = await optimizer.optimize(initial_agent, sampler)

    # 6. Inspect results
    for scored_agent in result.optimized_agents:
        print(f"Score: {scored_agent.overall_score:.3f}")
        print(f"Optimized instruction:\n{scored_agent.optimized_agent.instruction}")

asyncio.run(optimize_agent())
```

### Using a separate validation set

```python
# Separate validation set prevents overfitting to the training distribution
sampler_config = LocalEvalSamplerConfig(
    eval_config=eval_config,
    app_name="my_app",
    train_eval_set="train_cases",        # cases used during optimization iterations
    validation_eval_set="holdout_cases", # unseen cases for final scoring
)
```

### Extracting tool-call data for analysis

```python
# LocalEvalSampler.sample_and_score with capture_full_eval_data=True
# returns per-invocation tool call data in UnstructuredSamplingResult.data

result = await sampler.sample_and_score(
    candidate=my_agent,
    example_set="train",
    capture_full_eval_data=True,
)

for case_id, case_data in (result.data or {}).items():
    for invocation in case_data.get("invocations", []):
        for tool_call in invocation.get("tool_calls", []):
            print(f"  {tool_call['name']}({tool_call['args']}) → {tool_call['response']}")
```

---

## 9 · `SamplingResult` + `AgentWithScores` + `OptimizerResult` + `UnstructuredSamplingResult`

**Source:** `google.adk.optimization.data_types`

These four Pydantic models form the **data contract** between optimizers and samplers. Understanding their structure is essential for building custom optimizers, analysing results, and extending the optimization framework.

### `SamplingResult` — base scoring output

```python
from google.adk.optimization.data_types import SamplingResult

# scores: map from example_id → float score (higher is better, typically 0.0–1.0)
result = SamplingResult(scores={
    "case_001": 1.0,   # PASSED
    "case_002": 0.0,   # FAILED
    "case_003": 1.0,   # PASSED
})

avg_score = sum(result.scores.values()) / len(result.scores)
print(f"Average score: {avg_score:.2f}")  # 0.67
```

### `UnstructuredSamplingResult` — scores + raw evaluation data

```python
from google.adk.optimization.data_types import UnstructuredSamplingResult

# data: optional map from example_id → JSON-serializable evaluation info
# Typically contains: inputs, trajectories, tool calls, metric results
result = UnstructuredSamplingResult(
    scores={"case_001": 1.0, "case_002": 0.0},
    data={
        "case_001": {
            "invocations": [
                {
                    "user_prompt": "Book a flight to Paris",
                    "agent_response": "I've found 3 available flights.",
                    "tool_calls": [
                        {"name": "search_flights", "args": {"destination": "Paris"}, "response": {...}}
                    ],
                }
            ]
        }
    },
)
```

### `AgentWithScores` — optimized agent + score

```python
from google.adk.optimization.data_types import AgentWithScores
from google.adk.agents import LlmAgent

agent = LlmAgent(name="my_agent", model="gemini-2.5-flash", instruction="Be helpful.")

scored = AgentWithScores(
    optimized_agent=agent,
    overall_score=0.87,
)

# Extract the final agent and its validation score
print(f"Best prompt: {scored.optimized_agent.instruction}")
print(f"Validation score: {scored.overall_score:.2%}")  # 87.00%
```

### `OptimizerResult` — Pareto-front result set

```python
from google.adk.optimization.data_types import OptimizerResult, AgentWithScores

# OptimizerResult holds a *list* of agents because optimizers may return
# multiple Pareto-optimal solutions (e.g., tradeoffs between latency and quality)
result = OptimizerResult(
    optimized_agents=[
        AgentWithScores(optimized_agent=agent_a, overall_score=0.91),
        AgentWithScores(optimized_agent=agent_b, overall_score=0.88),
    ]
)

# Select the highest-scoring agent
best = max(result.optimized_agents, key=lambda a: a.overall_score or 0.0)
print(f"Selected agent score: {best.overall_score}")
```

### Extending `AgentWithScores` for multi-metric results

```python
from google.adk.optimization.data_types import AgentWithScores
from pydantic import Field
from typing import Optional
from google.adk.agents import LlmAgent

class MultiMetricAgentWithScores(AgentWithScores):
    """Custom scoring with separate latency and quality metrics."""
    quality_score: Optional[float] = Field(default=None)
    latency_score: Optional[float] = Field(default=None)  # lower latency = higher score

    @property
    def composite_score(self) -> float:
        q = self.quality_score or 0.0
        l = self.latency_score or 0.0
        return 0.7 * q + 0.3 * l  # weighted combination
```

---

## 10 · `ApiRegistry`

**Source:** `google.adk.integrations.api_registry.api_registry`

`ApiRegistry` connects to **Google Cloud API Registry** (Cloud API Hub) to discover registered MCP servers and expose them as `McpToolset` instances. This gives agents dynamic, governance-managed access to enterprise API surfaces: add a new MCP server to API Registry and agents automatically see it without code changes.

### Constructor — loads all MCP servers at init

```python
from google.adk.integrations.api_registry.api_registry import ApiRegistry

# Requires Application Default Credentials with roles/apihub.viewer
# On first call, fetches all mcpServers pages via the API Registry REST API
registry = ApiRegistry(
    api_registry_project_id="my-gcp-project",
    location="global",   # default
    header_provider=None,   # optional: Callable[[ReadonlyContext], dict[str, str]]
)
# registry._mcp_servers is now populated with server name → metadata
```

### `get_toolset` — create a `McpToolset` for a specific server

```python
from google.adk.integrations.api_registry.api_registry import ApiRegistry

registry = ApiRegistry(api_registry_project_id="my-gcp-project")

# Returns McpToolset pointing at the server's URL with auth headers
crm_toolset = registry.get_toolset(
    mcp_server_name="projects/my-gcp-project/locations/global/mcpServers/crm-server",
)

# Optional: filter to specific tools
analytics_toolset = registry.get_toolset(
    mcp_server_name="projects/my-gcp-project/locations/global/mcpServers/analytics-server",
    tool_filter=["get_metrics", "list_dashboards"],  # whitelist by name
    tool_name_prefix="analytics_",                   # prefix tool names
)
```

### Dynamic agent toolset discovery

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.integrations.api_registry.api_registry import ApiRegistry

async def build_dynamic_agent(project_id: str, mcp_server_name: str):
    """Builds an agent whose tools are discovered at runtime from API Registry."""
    registry = ApiRegistry(api_registry_project_id=project_id)

    # Toolset is resolved when agent runs — tools are fetched from the MCP server
    toolset = registry.get_toolset(mcp_server_name=mcp_server_name)

    agent = LlmAgent(
        name="enterprise_agent",
        model="gemini-2.5-pro",
        instruction="You have access to enterprise APIs. Use them to answer user queries.",
        tools=[toolset],
    )

    runner = Runner(
        agent=agent,
        app_name="enterprise_app",
        session_service=InMemorySessionService(),
    )
    return runner
```

### Adding per-request dynamic headers

```python
from google.adk.agents.readonly_context import ReadonlyContext

def user_context_headers(context: ReadonlyContext) -> dict[str, str]:
    """Injects the current user's tenant ID into every MCP call."""
    tenant_id = context.state.get("tenant_id", "default")
    return {"X-Tenant-ID": tenant_id}

registry = ApiRegistry(
    api_registry_project_id="my-gcp-project",
    header_provider=user_context_headers,
)

toolset = registry.get_toolset(
    mcp_server_name="projects/my-gcp-project/locations/global/mcpServers/billing-server",
)
# Each MCP call includes X-Tenant-ID from session state
```

### Listing all registered MCP servers

```python
from google.adk.integrations.api_registry.api_registry import ApiRegistry

registry = ApiRegistry(api_registry_project_id="my-gcp-project")

# _mcp_servers is a dict of server_name → server metadata
for server_name, server_meta in registry._mcp_servers.items():
    print(f"Server: {server_name}")
    if server_meta.get("urls"):
        print(f"  URL: {server_meta['urls'][0]}")
    if server_meta.get("displayName"):
        print(f"  Display: {server_meta['displayName']}")
```

---

## Volume index

| Vol. | Classes |
|------|---------|
| [1](./google_adk_class_deep_dives/) | `LlmAgent`, `RunConfig`, `Context`, `BasePlugin`, `App`, `Workflow`, `BaseNode`, `FunctionTool`, `RetryConfig`, `BaseTool` |
| [2](./google_adk_class_deep_dives_v2/) | `RemoteA2aAgent`, `LangGraphAgent`, `AuthCredential`, `GcsArtifactService`, `PubSubToolset`, `SpannerToolset` |
| [3](./google_adk_class_deep_dives_v3/) | `AgentTool`, `BuiltInPlanner`, `PlanReActPlanner`, `InMemorySessionService`, `VertexAiRagMemoryService`, `UnsafeLocalCodeExecutor`, `BuiltInCodeExecutor`, `LiteLlm`, `AgentEvaluator`, shell agents |
| [4](./google_adk_class_deep_dives_v4/) | `LongRunningFunctionTool`, `LiveRequestQueue`, `AnthropicLlm`/`Claude`, `SqliteSessionService`, `ExecuteBashTool`, `TransferToAgentTool`, `VertexAiMemoryBankService`, `SimplePromptOptimizer`, `SkillRegistry`/`SkillToolset`, `ToolConfirmation` |
| [5](./google_adk_class_deep_dives_v5/) | `VertexAiSessionService`, `VertexAiSearchTool`, `VertexAiCodeExecutor`, `APIHubToolset`, `ToolboxToolset`, `ConversationScenario`, `TrajectoryEvaluator`, `AuthConfig`/`AuthHandler`, `PreloadMemoryTool`, `CodeExecutorContext` |
| [6](./google_adk_class_deep_dives_v6/) | `ComputerUseTool`/`BaseComputer`, `OpenAPIToolset`/`RestApiTool`, `LlmEventSummarizer`, `Session`/`State`, `Event`/`EventActions`, `ExampleTool`, `GoogleSearchTool`/`UrlContextTool`, `LlmBackedUserSimulator`/`UserPersona`, `GEPARootAgentPromptOptimizer`, `EnvironmentSimulationPlugin` |
| [7](./google_adk_class_deep_dives_v7/) | `InvocationContext`, `SetModelResponseTool`, `DynamicNodeScheduler`/`ctx.run_node()`, retrieval tool chain, `FirestoreSessionService`/`FirestoreMemoryService`, `BigQueryToolset`/`WriteMode`, `LangchainTool`, `CrewaiTool`, `SlackRunner`, `FileArtifactService` |
| [8](./google_adk_class_deep_dives_v8/) | `ReadonlyContext`, `FunctionNode`, `JoinNode`/`Trigger`, `ContainerCodeExecutor`, `GkeCodeExecutor`, `AgentEngineSandboxCodeExecutor`, `ApplicationIntegrationToolset`, `BigtableToolset`, `OpenAILlm` |
| [9](./google_adk_class_deep_dives_v9/) | `Gemma`/`Gemma3Ollama`, `ContextCacheConfig`/`GeminiContextCacheManager`, `DataAgentToolset`, `DiscoveryEngineSearchTool`, `GoogleMapsGroundingTool`, `EnterpriseWebSearchTool`, `LoadMemoryTool`, `LoadArtifactsTool`, `exit_loop`/`get_user_choice_tool`, multi-turn eval suite |
| [10](./google_adk_class_deep_dives_v10/) | `Graph`/`Edge`, `NodeRunner`, `NodeState`/`NodeStatus`, `_ParallelWorker`, `ActiveStreamingTool`/`TranscriptionEntry`, `CachePerformanceAnalyzer`, `StreamingResponseAggregator`, `AgentRefConfig`/`ArgumentConfig`/`CodeConfig`, `BaseAuthProvider`/`AuthProviderRegistry`, `FinishTaskTool`/`TaskRequest`/`TaskResult` |
| [11](./google_adk_class_deep_dives_v11/) | `GoogleApiToolset` family, `load_web_page`, `UiWidget`, `_ToolNode`, `SqliteSpanExporter`, `RougeEvaluator`/`FinalResponseMatchV1`, `FinalResponseMatchV2Evaluator`, `HallucinationsV1Evaluator`, function calling pipeline, `_ContentLlmRequestProcessor`+`_InstructionsLlmRequestProcessor` |
| [12](./google_adk_class_deep_dives_v12/) | `BasePlugin`/`PluginManager`, `ContextFilterPlugin`, `ReflectAndRetryToolPlugin`, `GlobalInstructionPlugin`, `SaveFilesAsArtifactsPlugin`, `MultimodalToolResultsPlugin`, `DebugLoggingPlugin`, `RunConfig`/`StreamingMode`/`ToolThreadPoolConfig`, `AuthenticatedFunctionTool`, `FeatureName`/feature flags |
| [13](./google_adk_class_deep_dives_v13/) | `ApigeeLlm`/`ApiType`, `AudioCacheManager`/`AudioCacheConfig`, `AudioTranscriber`, `TranscriptionManager`, `AgentEngineSandboxComputer`/VMaaS, `ParameterManagerClient`, `SecretManagerClient`, `GcpAuthProvider`/`GcpAuthProviderScheme`, `AgentRegistry`/`AgentRegistrySingleMcpToolset`, `OAuth2CredentialRefresher` pipeline |
| [14](./google_adk_class_deep_dives_v14/) | `A2aAgentExecutor`/`A2aAgentExecutorConfig`/`ExecuteInterceptor`, `Context`/`ToolContext` advanced API, `GcsArtifactService` deep-dive, `LlmEventSummarizer`/`BaseEventsSummarizer`, `EventsCompactionConfig` advanced, `SpannerVectorStoreSettings`, `LangGraphAgent`+checkpointer, `Trigger` sub-branch/isolation, `ResumabilityConfig`, `PubSubToolset` advanced |
| **15** | **`LlmRequest`, `LlmResponse`, `BaseLlm`, `LLMRegistry`, `BaseLlmFlow`/`AutoFlow`/`SingleFlow`, `BaseLlmRequestProcessor`/`BaseLlmResponseProcessor`, `Sampler`/`AgentOptimizer`, `LocalEvalSampler`/`LocalEvalSamplerConfig`, optimization data types, `ApiRegistry`** |

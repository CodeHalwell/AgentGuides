---
title: "PydanticAI: 10 Source-Verified Class Deep Dives (2.55.0)"
description: "Runnable, source-verified code examples for FallbackModel (rewind/ResponseHandler), InstrumentedModel/InstrumentationSettings, Caching, LocalWorkspace, ResolveModelId, SetMetadataToolset, WrapperToolset, OpenAIEmbeddingModel, ImageGeneration, and Hooks (wrap-style) — verified against pydantic-ai 2.55.0."
framework: pydanticai
language: python
sidebar:
  order: 131
---

# 10 Source-Verified Class Deep Dives — v2.55.0

Verified against **pydantic-ai 2.55.0** (installed package, sources read directly).
Modules consulted: `pydantic_ai/models/fallback.py`, `pydantic_ai/models/instrumented.py`,
`pydantic_ai/capabilities/caching.py`, `pydantic_ai/capabilities/local_workspace.py`,
`pydantic_ai/capabilities/resolve_model_id.py`, `pydantic_ai/toolsets/set_metadata.py`,
`pydantic_ai/toolsets/wrapper.py`, `pydantic_ai/embeddings/openai.py`,
`pydantic_ai/capabilities/image_generation.py`, `pydantic_ai/capabilities/hooks.py`.

```bash
pip install "pydantic-ai==2.55.0"
python -c "import pydantic_ai; print(pydantic_ai.__version__)"
#> 2.55.0
```

---

## 1. `FallbackModel` — rewind, `ResponseHandler`, and `FallbackOn`

**Module:** `pydantic_ai.models.fallback`

`FallbackModel` chains models so that when one fails (or its response is rejected) the next model
in the list is tried. In 2.55.0 the `FallbackOn` type accepts three distinct shapes:

| `fallback_on` element | What it matches |
|---|---|
| `type[Exception]` (e.g. `ModelAPIError`) | Any instance of that exception class |
| `ExceptionHandler` callable `(exc) → bool` | Arbitrary predicate on the exception |
| `ResponseHandler` callable `(ModelResponse) → bool` | Reject a successful response by content |

`ResponseHandler` is the new capability added in 2.x — the rewind mechanism replaces the previous
response with the next model's answer transparently, so callers never see the intermediate failure.

### Constructor (from source)

```python
class FallbackModel(Model):
    def __init__(
        self,
        default_model: Model | KnownModelName | str,
        *fallback_models: Model | KnownModelName | str,
        fallback_on: FallbackOn = (ModelAPIError,),
    ): ...
```

`FallbackOn` expands to:

```python
FallbackOn = (
    ResponseHandler                                      # single callable
    | Sequence[type[Exception] | ExceptionHandler | ResponseHandler]  # mixed list
)
```

### Example 1 — exception-based fallback

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.test import TestModel

# Primary always raises; fallback returns cleanly.
def always_fail(*args, **kwargs):
    raise ModelAPIError("primary is down")

fallback = FallbackModel(
    TestModel(custom_result_text="primary answer"),   # primary
    TestModel(custom_result_text="fallback answer"),  # fallback
    fallback_on=(ModelAPIError,),
)
agent = Agent(fallback)

async def main():
    result = await agent.run("ping")
    print(result.data)  # "primary answer" (TestModel never raises)

asyncio.run(main())
```

### Example 2 — `ResponseHandler`: reject low-confidence responses

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ModelResponse

def reject_uncertain(response: ModelResponse) -> bool:
    """Return True to trigger fallback (i.e. reject this response)."""
    for part in response.parts:
        if hasattr(part, "content") and "I don't know" in (part.content or ""):
            return True
    return False

uncertain_model = TestModel(custom_result_text="I don't know the answer")
confident_model = TestModel(custom_result_text="The capital is Paris.")

fallback = FallbackModel(
    uncertain_model,
    confident_model,
    fallback_on=reject_uncertain,
)
agent = Agent(fallback)

async def main():
    result = await agent.run("What is the capital of France?")
    # The uncertain_model's response is rejected, so confident_model responds.
    print(result.data)
    # Inspect which model actually answered via model_name on the usage
    print(result.usage())

asyncio.run(main())
```

### Example 3 — mixed `fallback_on` list

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.test import TestModel

def is_rate_limit(exc: Exception) -> bool:
    return isinstance(exc, ModelAPIError) and "429" in str(exc)

def is_empty_response(response: ModelResponse) -> bool:
    return all(not getattr(p, "content", None) for p in response.parts)

fallback = FallbackModel(
    TestModel(custom_result_text="primary"),
    TestModel(custom_result_text="secondary"),
    TestModel(custom_result_text="tertiary"),
    fallback_on=[
        ModelAPIError,        # exception type
        is_rate_limit,        # exception handler callable
        is_empty_response,    # response handler callable
    ],
)

agent = Agent(fallback)

async def main():
    result = await agent.run("hello")
    print(result.data)

asyncio.run(main())
```

---

## 2. `InstrumentedModel` & `InstrumentationSettings`

**Module:** `pydantic_ai.models.instrumented`

`InstrumentedModel` wraps any `Model` and emits OpenTelemetry spans for every request.
`InstrumentationSettings` controls what gets recorded. The convenience function `instrument_model`
avoids double-wrapping.

### Constructor / fields (from source)

```python
@dataclass(init=False)
class InstrumentationSettings:
    tracer: Tracer
    include_binary_content: bool   # default True — embed BinaryContent in spans
    include_content: bool          # default True — embed prompts & completions
    include_model_request_parameters: bool  # default True — embed full tool schemas
    version: Literal[2, 3, 4, 5, 6]        # default 5 (latest stable)
    use_aggregated_usage_attribute_names: bool  # default True

    def __init__(
        self,
        *,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
        include_binary_content: bool = True,
        include_content: bool = True,
        include_model_request_parameters: bool = True,
        version: Literal[2, 3, 4, 5, 6] = 5,
        use_aggregated_usage_attribute_names: bool = True,
    ): ...


def instrument_model(model: Model, instrument: InstrumentationSettings | bool) -> Model:
    """Wrap model in InstrumentedModel; no-op when already wrapped."""
```

### Example 1 — instrument a model directly

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentedModel, InstrumentationSettings
from pydantic_ai.models.test import TestModel

settings = InstrumentationSettings(
    include_content=True,
    include_binary_content=False,  # skip large binary payloads in spans
)
model = InstrumentedModel(TestModel(custom_result_text="hello"), settings)

agent = Agent(model)

async def main():
    result = await agent.run("say hi")
    print(result.data)

asyncio.run(main())
```

### Example 2 — Logfire integration (the standard path)

```python
# Logfire sets the global OTel tracer provider, so instrument=True picks it up.
# Run:  pip install logfire  &&  logfire auth

import logfire
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings, instrument_model
from pydantic_ai.models.test import TestModel

logfire.configure()

model = instrument_model(TestModel(custom_result_text="traced!"), instrument=True)
agent = Agent(model)

async def main():
    result = await agent.run("trace me")
    print(result.data)

asyncio.run(main())
```

### Example 3 — suppress content from spans (privacy mode)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentedModel, InstrumentationSettings
from pydantic_ai.models.test import TestModel

# Spans contain timing and token counts, but no prompt/response text.
settings = InstrumentationSettings(
    include_content=False,
    include_binary_content=False,
    include_model_request_parameters=False,
)
agent = Agent(InstrumentedModel(TestModel(custom_result_text="private"), settings))

async def main():
    result = await agent.run("confidential question")
    print(result.data)

asyncio.run(main())
```

---

## 3. `Caching` — cross-provider prompt caching capability

**Module:** `pydantic_ai.capabilities.caching`

`Caching` injects `ModelSettings.cache` into every run so prompt caching is enabled without
touching each call site. It works portably across providers; on models that cache implicitly
(e.g. some Gemini tiers) it has no effect.

### Class (from source)

```python
@dataclass
class Caching(AbstractCapability[Any]):
    retention: bool | CacheRetention = True
    # True   → provider default retention
    # False  → disable library-managed caching
    # '5m' | '30m' | '1h' → specific tier, snapped to nearest provider support

    messages: bool = True
    # True  → also cache the growing conversation
    # False → cache only stable prefix (instructions + tool schemas)

    id: str | None = 'caching'
    # Fixed so two Caching instances resolve to one; pass id=None for derived ids.
```

### Example 1 — minimal: enable caching globally

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.caching import Caching
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text="cached answer"),
    capabilities=[Caching()],           # enable with provider defaults
)

async def main():
    r1 = await agent.run("What is 2+2?")
    r2 = await agent.run("What is 2+2?", message_history=r1.new_messages())
    print(r1.data, r2.data)

asyncio.run(main())
```

### Example 2 — 1-hour retention, stable prefix only

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.caching import Caching
from pydantic_ai.models.test import TestModel

# Long-lived instructions (system prompt + tools) cached for 1 hour.
# Message history NOT cached — suitable for many short one-off sessions.
agent = Agent(
    TestModel(custom_result_text="ok"),
    system_prompt="You are a helpful assistant with 10 000 words of context...",
    capabilities=[Caching(retention='1h', messages=False)],
)

async def main():
    # Each run gets a fresh conversation but the stable prefix is cached.
    for prompt in ["ping", "pong", "hello"]:
        r = await agent.run(prompt)
        print(r.data)

asyncio.run(main())
```

### Example 3 — disable caching for a specific run

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.caching import Caching
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text="result"),
    capabilities=[Caching(retention='30m')],
)

async def main():
    # Normal run — caching applied by the Caching capability.
    cached = await agent.run("query A")

    # Override caching off for this single call.
    uncached = await agent.run(
        "query B",
        model_settings=ModelSettings(cache=False),
    )
    print(cached.data, uncached.data)

asyncio.run(main())
```

---

## 4. `LocalWorkspace` — giving agents a real filesystem

**Module:** `pydantic_ai.capabilities.local_workspace`

`LocalWorkspace` mounts a directory on the host machine as `ctx.workspace` inside every tool call.
Tools can then read, write, and execute commands inside that directory. Nothing is sandboxed — this
is for trusted, local workflows. Use a container backend for untrusted code.

### Class (from source)

```python
@dataclass
class LocalWorkspace(AbstractCapability[AgentDepsT]):
    working_dir: str | Path
    # Where commands start and relative paths resolve.
    # '~' is expanded; '.' is today's cwd.

    read_only: bool = False
    # Wrap in ReadOnlyWorkspace — allows reads but blocks writes.

    env: Mapping[str, str] | None = None
    # Extra env vars for every command (on top of PATH, HOME, LANG).
    # Does NOT inherit os.environ, so secrets don't leak.

    id: str | None = 'local_workspace'
    # Fixed so a later instance replaces the earlier one wholesale.
```

### Example 1 — read a file from the workspace

```python
import asyncio
import tempfile
from pathlib import Path
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.local_workspace import LocalWorkspace
from pydantic_ai.models.test import TestModel

async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Seed a file in the workspace.
        Path(tmpdir, "readme.txt").write_text("Hello from the workspace!")

        agent = Agent(
            TestModel(custom_result_text="done"),
            capabilities=[LocalWorkspace(tmpdir)],
        )

        @agent.tool
        async def read_readme(ctx: RunContext[None]) -> str:
            ws = ctx.workspace
            content = await ws.read_text("readme.txt")
            return content

        result = await agent.run("read readme")
        print(result.data)

asyncio.run(main())
```

### Example 2 — write a file from a tool

```python
import asyncio
import tempfile
from pathlib import Path
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.local_workspace import LocalWorkspace
from pydantic_ai.models.test import TestModel

async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = Agent(
            TestModel(custom_result_text="wrote"),
            capabilities=[LocalWorkspace(tmpdir)],
        )

        @agent.tool
        async def write_report(ctx: RunContext[None], content: str) -> str:
            await ctx.workspace.write_text("report.md", content)
            return "written"

        result = await agent.run("write the report")
        print(Path(tmpdir, "report.md").read_text())

asyncio.run(main())
```

### Example 3 — read-only workspace guards against accidental writes

```python
import asyncio
import tempfile
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.local_workspace import LocalWorkspace
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import WorkspaceError

async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = Agent(
            TestModel(custom_result_text="blocked"),
            capabilities=[LocalWorkspace(tmpdir, read_only=True)],
        )

        @agent.tool
        async def try_write(ctx: RunContext[None]) -> str:
            try:
                await ctx.workspace.write_text("secret.txt", "oops")
                return "wrote"
            except (WorkspaceError, PermissionError, Exception) as e:
                return f"blocked: {type(e).__name__}"

        result = await agent.run("try to write")
        print(result.data)

asyncio.run(main())
```

---

## 5. `ResolveModelId` — dynamic model routing

**Module:** `pydantic_ai.capabilities.resolve_model_id`

`ResolveModelId` intercepts every model-ID resolution step. The resolver receives a
`ModelResolutionContext` and the string model ID; it returns a `Model` instance or `None` to pass
control to the next capability (or built-in `infer_model`).

### Class (from source)

```python
ModelIdResolver = (
    Callable[[ModelResolutionContext[AgentDepsT], str], Model | None]
    | Callable[[ModelResolutionContext[AgentDepsT], str], Awaitable[Model | None]]
)

@dataclass
class ResolveModelId(AbstractCapability[AgentDepsT]):
    resolver: ModelIdResolver[AgentDepsT]
```

### Example 1 — route by model-name prefix

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.resolve_model_id import ResolveModelId
from pydantic_ai.models.test import TestModel
from pydantic_ai.models import ModelResolutionContext

def route_model(ctx: ModelResolutionContext, model_id: str):
    """Map custom short-names to concrete models."""
    table = {
        "fast": TestModel(custom_result_text="fast answer"),
        "smart": TestModel(custom_result_text="smart answer"),
    }
    return table.get(model_id)  # None → fall through to infer_model

agent = Agent(
    "fast",   # will be resolved by our capability
    capabilities=[ResolveModelId(route_model)],
)

async def main():
    result = await agent.run("hello")
    print(result.data)   # "fast answer"

asyncio.run(main())
```

### Example 2 — async resolver fetching model config from a registry

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.resolve_model_id import ResolveModelId
from pydantic_ai.models.test import TestModel
from pydantic_ai.models import ModelResolutionContext

# Simulated registry lookup (real code might call a DB or config service).
_REGISTRY: dict[str, str] = {
    "prod-agent": "smart",
    "dev-agent": "fast",
}

async def async_resolver(ctx: ModelResolutionContext, model_id: str):
    canonical = _REGISTRY.get(model_id)
    if canonical == "smart":
        return TestModel(custom_result_text="smart")
    if canonical == "fast":
        return TestModel(custom_result_text="fast")
    return None   # pass through

agent = Agent("prod-agent", capabilities=[ResolveModelId(async_resolver)])

async def main():
    result = await agent.run("query")
    print(result.data)   # "smart"

asyncio.run(main())
```

### Example 3 — per-run model selection via deps

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.resolve_model_id import ResolveModelId
from pydantic_ai.models.test import TestModel
from pydantic_ai.models import ModelResolutionContext

@dataclass
class Deps:
    prefer_fast: bool

def deps_driven_resolver(ctx: ModelResolutionContext[Deps], model_id: str):
    if model_id != "auto":
        return None
    if ctx.deps and ctx.deps.prefer_fast:
        return TestModel(custom_result_text="fast model chosen")
    return TestModel(custom_result_text="smart model chosen")

agent = Agent("auto", deps_type=Deps, capabilities=[ResolveModelId(deps_driven_resolver)])

async def main():
    r1 = await agent.run("hi", deps=Deps(prefer_fast=True))
    r2 = await agent.run("hi", deps=Deps(prefer_fast=False))
    print(r1.data, "|", r2.data)

asyncio.run(main())
```

---

## 6. `SetMetadataToolset` — merging metadata onto tool definitions

**Module:** `pydantic_ai.toolsets.set_metadata`

`SetMetadataToolset` wraps any toolset and shallow-merges a `dict` of metadata onto every
`ToolDefinition` before it reaches the model. This is useful for tagging tools with routing hints,
cost tiers, or capability markers without modifying the tool's implementation.

### Class (from source)

```python
@dataclass(init=False)
class SetMetadataToolset(PreparedToolset[AgentDepsT]):
    prepare_func: ToolsPrepareFunc[AgentDepsT]  # auto-generated in __init__
    metadata: dict[str, Any]

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        metadata: dict[str, Any],
    ) -> None: ...
```

The implementation shallow-merges: `{**(td.metadata or {}), **self.metadata}` — your keys
win over any pre-existing ones.

### Example 1 — tag tools with a cost tier

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.set_metadata import SetMetadataToolset
from pydantic_ai.models.test import TestModel

toolset = FunctionToolset()

@toolset.tool
def expensive_lookup(query: str) -> str:
    return f"result for {query}"

# Tag all tools in the toolset as "expensive".
tagged = SetMetadataToolset(toolset, {"cost_tier": "expensive", "owner": "search-team"})

agent = Agent(TestModel(custom_result_text="done"), toolsets=[tagged])

async def main():
    result = await agent.run("look up something")
    print(result.data)

asyncio.run(main())
```

### Example 2 — stack with `FilteredToolset` to gate by metadata

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.set_metadata import SetMetadataToolset
from pydantic_ai.toolsets.filtered import FilteredToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.models.test import TestModel

toolset = FunctionToolset()

@toolset.tool
def free_tool(query: str) -> str:
    return f"free: {query}"

@toolset.tool
def paid_tool(query: str) -> str:
    return f"paid: {query}"

# Tag all tools as "free" first, then override paid_tool.
free_tagged = SetMetadataToolset(toolset, {"tier": "free"})

# FilteredToolset downstream can now inspect metadata.
def allow_free_only(ctx: RunContext, tool_def: ToolDefinition) -> bool:
    return (tool_def.metadata or {}).get("tier") == "free"

gated = FilteredToolset(free_tagged, allow_free_only)

agent = Agent(TestModel(custom_result_text="ok"), toolsets=[gated])

async def main():
    result = await agent.run("use tools")
    print(result.data)

asyncio.run(main())
```

---

## 7. `WrapperToolset` — base class for custom toolset wrappers

**Module:** `pydantic_ai.toolsets.wrapper`

`WrapperToolset` is the recommended base for any toolset that wraps another. It delegates all
lifecycle methods (`for_run`, `for_run_step`, `__aenter__`, `__aexit__`, `get_tools`,
`call_tool`, `apply`, `visit_and_replace`, `get_instructions`) to `self.wrapped`, so you only
override the methods you actually need.

### Key interface (from source)

```python
@dataclass
class WrapperToolset(AbstractToolset[AgentDepsT]):
    wrapped: AbstractToolset[AgentDepsT]

    # Override any of:
    async def get_tools(self, ctx) -> dict[str, ToolsetTool]: ...
    async def call_tool(self, name, tool_args, ctx, tool) -> Any: ...
    async def for_run(self, ctx) -> AbstractToolset: ...     # once per run
    async def for_run_step(self, ctx) -> AbstractToolset: ...  # once per step
```

### Example 1 — audit-logging wrapper

```python
import asyncio
from dataclasses import dataclass
from typing import Any
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.models.test import TestModel

@dataclass
class AuditToolset(WrapperToolset):
    """Log every tool call and its return value."""

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext,
        tool: ToolsetTool,
    ) -> Any:
        print(f"[AUDIT] calling {name!r} with {tool_args}")
        result = await super().call_tool(name, tool_args, ctx, tool)
        print(f"[AUDIT] {name!r} returned {result!r}")
        return result


toolset = FunctionToolset()

@toolset.tool
def add(a: int, b: int) -> int:
    return a + b

audited = AuditToolset(wrapped=toolset)
agent = Agent(TestModel(custom_result_text="computed"), toolsets=[audited])

async def main():
    result = await agent.run("add 2 and 3")
    print(result.data)

asyncio.run(main())
```

### Example 2 — per-run context injection into tools

```python
import asyncio
from dataclasses import dataclass, replace
from typing import Any
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.models.test import TestModel

@dataclass
class TimestampedToolset(WrapperToolset):
    """Injects a timestamp into every tool's return value."""

    import datetime

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext,
        tool: ToolsetTool,
    ) -> Any:
        import datetime
        raw = await super().call_tool(name, tool_args, ctx, tool)
        ts = datetime.datetime.now(datetime.UTC).isoformat()
        return {"result": raw, "timestamp": ts}


toolset = FunctionToolset()

@toolset.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

timestamped = TimestampedToolset(wrapped=toolset)
agent = Agent(TestModel(custom_result_text="greeted"), toolsets=[timestamped])

async def main():
    result = await agent.run("greet Alice")
    print(result.data)

asyncio.run(main())
```

---

## 8. `OpenAIEmbeddingModel` — semantic embeddings

**Module:** `pydantic_ai.embeddings.openai`

`OpenAIEmbeddingModel` implements the `EmbeddingModel` ABC for OpenAI's embeddings API (and any
OpenAI-compatible provider). Use it to embed text for similarity search, RAG pipelines,
or semantic clustering.

### Constructor (from source)

```python
@dataclass(init=False)
class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_name: OpenAIEmbeddingModelName,  # str | LatestOpenAIEmbeddingModelNames
        *,
        provider: OpenAIEmbeddingsCompatibleProvider | Literal['openai'] | Provider[AsyncOpenAI] = 'openai',
        settings: EmbeddingSettings | None = None,
    ): ...

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,  # 'query' | 'document'
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult: ...
```

### Example 1 — embed a single document

```python
import asyncio
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel

# Requires OPENAI_API_KEY.
async def main():
    model = OpenAIEmbeddingModel("text-embedding-3-small")
    result = await model.embed("Hello, world!", input_type="document")
    print(f"Dims: {len(result.embeddings[0])}")  # 1536 for text-embedding-3-small
    print(f"Tokens used: {result.usage}")

asyncio.run(main())
```

### Example 2 — batch embed and cosine similarity

```python
import asyncio
import math
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel

def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(x**2 for x in b))
    return dot / (norm_a * norm_b)

# Requires OPENAI_API_KEY.
async def main():
    model = OpenAIEmbeddingModel("text-embedding-3-small")

    docs = [
        "Pydantic AI is an agent framework.",
        "Python is a programming language.",
        "Cats are mammals.",
    ]
    query = "What is pydantic-ai?"

    doc_result = await model.embed(docs, input_type="document")
    q_result = await model.embed(query, input_type="query")

    q_vec = q_result.embeddings[0]
    for doc, d_vec in zip(docs, doc_result.embeddings):
        score = cosine_sim(q_vec, d_vec)
        print(f"{score:.3f}  {doc[:50]}")

asyncio.run(main())
```

### Example 3 — OpenAI-compatible provider (e.g. Azure)

```python
import asyncio
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.azure import AzureProvider

async def main():
    provider = AzureProvider(
        azure_endpoint="https://my-resource.openai.azure.com",
        api_version="2024-02-01",
        api_key="<azure-key>",
    )
    model = OpenAIEmbeddingModel("text-embedding-3-small", provider=provider)
    result = await model.embed(["Azure embeddings!"], input_type="document")
    print(len(result.embeddings[0]))

asyncio.run(main())
```

---

## 9. `ImageGeneration` capability — native and fallback image generation

**Module:** `pydantic_ai.capabilities.image_generation`

`ImageGeneration` is the capability form of image generation. It automatically uses the model's
**native** image generation when the model supports it (e.g., GPT-Image-1, Gemini Imagen),
and falls back to a separate image model or a subagent when the primary model doesn't support it.

### Key fields (from source)

```python
class ImageGeneration(NativeOrLocalTool[AgentDepsT]):
    local: str | ImageGenerator | Tool | Callable | AbstractToolset | bool | None = None
    # Local fallback: an ImageGenerator, a custom Tool, or a callable.

    fallback_image_model: ImageGenerationModel | KnownModelName | str | None = None
    # A dedicated image-gen model for providers that don't support native gen.

    fallback_subagent_model: Model | KnownModelName | str | None = None
    # An image-capable chat model; runs a sub-agent that calls the native ImageGenerationTool.

    # Portable geometry settings (applied to direct fallback):
    dimensions: ImageDimensions | None = None
    aspect_ratio: ImageGenerationAspectRatio | None = None

    # Passed to the native ImageGenerationTool:
    quality: str | None = None
    style: str | None = None
```

### Example 1 — enable native image generation (model that supports it)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.image_generation import ImageGeneration
from pydantic_ai.models.test import TestModel

# On a real model (e.g. 'openai:gpt-image-1') this triggers native gen.
# TestModel here just shows the wiring.
agent = Agent(
    TestModel(custom_result_text="image generated"),
    capabilities=[ImageGeneration()],
)

async def main():
    result = await agent.run("Generate an image of a sunset over the ocean.")
    print(result.data)

asyncio.run(main())
```

### Example 2 — explicit fallback image model

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.image_generation import ImageGeneration
from pydantic_ai.models.test import TestModel

# When the primary model can't generate images, use dall-e-3 directly.
# Requires OPENAI_API_KEY in real usage.
agent = Agent(
    TestModel(custom_result_text="via fallback"),
    capabilities=[
        ImageGeneration(
            fallback_image_model="openai:dall-e-3",
            dimensions=(1024, 1024),
        )
    ],
)

async def main():
    result = await agent.run("A photo-realistic cat.")
    print(result.data)

asyncio.run(main())
```

### Example 3 — subagent fallback for chat models without native image gen

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.image_generation import ImageGeneration
from pydantic_ai.models.test import TestModel

# The subagent runs on gpt-4o which calls ImageGenerationTool natively.
# Requires OPENAI_API_KEY in real usage.
agent = Agent(
    TestModel(custom_result_text="subagent result"),
    capabilities=[
        ImageGeneration(
            fallback_subagent_model="openai:gpt-4o",
            aspect_ratio="16:9",
        )
    ],
)

async def main():
    result = await agent.run("A wide landscape painting.")
    print(result.data)

asyncio.run(main())
```

---

## 10. `Hooks` — wrap-style lifecycle callbacks

**Module:** `pydantic_ai.capabilities.hooks`

`Hooks` provides a decorator-based API for hooking into any step of an agent run. In addition
to simple before/after hooks, 2.55.0 offers **wrap-style** hooks that are async generators:
they `yield` to let the normal execution happen, then run cleanup on the other side.

### Key hook types (from source)

| Hook | Signature shape | When it fires |
|---|---|---|
| `before_run` | `(ctx, prompt) → None` | Before the first node in a run |
| `after_run` | `(ctx, result) → None` | After the last node succeeds |
| `run` | async generator wrapping the whole run | Surrounds the entire run |
| `before_model_request` | `(ctx, request_ctx) → request_ctx` | Before each model API call |
| `after_model_response` | `(ctx, response) → response` | After each model response |
| `before_tool_execute` | `(ctx, name, args) → args` | Before a tool function executes |
| `after_tool_execute` | `(ctx, name, result) → result` | After a tool function returns |
| `run_error` | `(ctx, exc) → None` | When the run raises an unhandled exception |

### Example 1 — decorator registration: log every model request

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.models.test import TestModel

hooks = Hooks()

@hooks.on.before_model_request
async def log_request(ctx, request_context):
    print(f"[HOOK] model request: usage so far = {ctx.usage}")
    return request_context  # must return (possibly modified) request_context

agent = Agent(TestModel(custom_result_text="hi"), capabilities=[hooks])

async def main():
    result = await agent.run("hello")
    print(result.data)

asyncio.run(main())
```

### Example 2 — wrap-style hook: timer around each run

```python
import asyncio
import time
from collections.abc import AsyncIterator
from pydantic_ai import Agent
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.models.test import TestModel

hooks = Hooks()

@hooks.on.run
async def timer(ctx, handler) -> AsyncIterator[None]:
    start = time.perf_counter()
    try:
        yield   # execution happens here
    finally:
        elapsed = time.perf_counter() - start
        print(f"[TIMER] run took {elapsed:.3f}s")

agent = Agent(TestModel(custom_result_text="timed"), capabilities=[hooks])

async def main():
    result = await agent.run("time me")
    print(result.data)

asyncio.run(main())
```

### Example 3 — constructor kwargs: before/after tool execute

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.models.test import TestModel

call_log: list[str] = []

async def before_tool(ctx: RunContext, tool_name: str, tool_args: dict) -> dict:
    call_log.append(f"before:{tool_name}")
    return tool_args   # must return args (possibly modified)

async def after_tool(ctx: RunContext, tool_name: str, result) -> object:
    call_log.append(f"after:{tool_name}={result!r}")
    return result   # must return result (possibly modified)

hooks = Hooks(
    before_tool_execute=before_tool,
    after_tool_execute=after_tool,
)

agent = Agent(
    TestModel(custom_result_text="done"),
    capabilities=[hooks],
)

@agent.tool
def double(n: int) -> int:
    return n * 2

async def main():
    result = await agent.run("double 5")
    print(result.data)
    print(call_log)   # ['before:double', 'after:double=10']

asyncio.run(main())
```

---

## Cross-references

| Class / concept | Previous deep dive |
|---|---|
| `FilteredToolset` | [v2.54.0 deep dives](./pydantic_ai_class_deep_dives_v2_54/) |
| `PreparedToolset` / `DynamicToolset` | [v2.54.0 deep dives](./pydantic_ai_class_deep_dives_v2_54/) |
| `Thinking` / `WebSearch` capabilities | [v2.54.0 deep dives](./pydantic_ai_class_deep_dives_v2_54/) |
| `ApprovalRequiredToolset` | [v2.51.0 deep dives](./pydantic_ai_class_deep_dives_v2_51/) |
| `DeferredLoadingToolset` | [v2.51.0 deep dives](./pydantic_ai_class_deep_dives_v2_51/) |
| `FallbackModel` (response handlers) | [v2.46.0 deep dives](./pydantic_ai_class_deep_dives_v2_46/) |
| `CombinedToolset` | [v2.46.0 deep dives](./pydantic_ai_class_deep_dives_v2_46/) |
| `RealtimeSession` / `RealtimeModelSettings` | [v2.40.0 deep dives](./pydantic_ai_class_deep_dives_v2_40/) |
| `Agent` / `RunContext` / `UsageLimits` | [Aug 2026 deep dives (v2.33.0)](./pydantic_ai_class_examples_2026_08/) |
| Full class & API reference | [Comprehensive guide](./pydantic_ai_comprehensive_guide/#class--api-reference) |

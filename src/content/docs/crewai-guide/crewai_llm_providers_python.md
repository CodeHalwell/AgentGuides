---
title: "LLM providers and routing"
description: "How crewai.LLM picks a native SDK vs LiteLLM, the full list of supported providers, reasoning/thinking parameters, and building a custom BaseLLM."
framework: crewai
language: python
sidebar:
  label: "LLM providers"
  order: 25
---

> **Verified against crewai==1.14.3a2** (source: `crewai/llm.py`, `crewai/llms/base_llm.py`, `crewai/llms/providers/`).

The `LLM` factory is doing more than it looks. It's a `__new__`-time router that picks between a native provider SDK and the LiteLLM fallback based on the model name — and it's stricter than older docs suggest.

## Minimal runnable example

```python
from crewai import LLM

# Native provider (OpenAI) — uses `openai` SDK directly
gpt = LLM(model="openai/gpt-4o-mini", temperature=0.2)

# Provider prefix omitted — LLM infers from model name
gpt = LLM(model="gpt-4o-mini")

# Non-native → LiteLLM fallback (requires litellm installed)
mistral = LLM(model="mistral/mistral-large-latest")
```

The first argument **must** include a recognised provider prefix if the model name is ambiguous; otherwise `LLM._infer_provider_from_model` tries to guess.

## How routing works

```text
  LLM(model=..., **kwargs)
            │
            ▼
  ┌─────────────────────────────────────┐
  │ 1. kwargs has `provider=...`        │
  │    → force that provider, native    │
  └─────────────────────────────────────┘
            │ no
            ▼
  ┌─────────────────────────────────────┐
  │ 2. "/" in model ("openai/gpt-4o")   │
  │    → prefix looked up in            │
  │      provider_mapping               │
  │    → if model is in the native      │
  │      constants → native SDK         │
  │    → otherwise → LiteLLM            │
  └─────────────────────────────────────┘
            │ no
            ▼
  ┌─────────────────────────────────────┐
  │ 3. No "/" — infer from name pattern │
  │    (gpt-*/claude-*/gemini-*…)       │
  └─────────────────────────────────────┘
            │
            ▼
  Native SDK? → provider module under
                crewai.llms.providers/
  LiteLLM? → litellm.completion(...)
```

Native providers (source: `SUPPORTED_NATIVE_PROVIDERS`):

| Prefix | Canonical provider | SDK package |
|---|---|---|
| `openai` | openai | `openai` |
| `anthropic`, `claude` | anthropic | `anthropic` |
| `azure`, `azure_openai` | azure | `openai` (Azure) |
| `google`, `gemini` | gemini | `google-generativeai` |
| `bedrock`, `aws` | bedrock | `boto3` |
| `openrouter` | openrouter | `openai`-compatible |
| `deepseek`, `ollama`, `ollama_chat`, `hosted_vllm`, `cerebras`, `dashscope` | same | `openai`-compatible |

Anything else (`mistral/`, `groq/`, `cohere/`, custom prefixes) falls through to LiteLLM.

## Force LiteLLM

```python
LLM(model="openai/gpt-4o-mini", is_litellm=True)
```

Handy when you want LiteLLM's uniform interface even for natively supported models (e.g. to reuse a LiteLLM router config).

## Constructor fields (litellm-backed `LLM`)

The default `LLM` class extends `BaseLLM` with LiteLLM-specific knobs:

| Field | Type | Notes |
|---|---|---|
| `model` | `str` | Required. `provider/model` or a bare name. |
| `temperature` | `float \| None` | |
| `top_p`, `top_logprobs`, `logprobs` | various | Standard sampling params. |
| `max_tokens`, `max_completion_tokens` | `int \| float \| None` | |
| `response_format` | `JsonResponseFormat \| type[BaseModel] \| None` | Provider-native structured output. |
| `stop` | `list[str]` | Stop sequences. |
| `seed` | `int \| None` | Deterministic sampling where supported. |
| `presence_penalty`, `frequency_penalty`, `logit_bias` | `float` / `dict` | OpenAI-style knobs. |
| `base_url` | `str \| None` | Inherited from `BaseLLM`. Used by native OpenAI-compatible paths (`ollama/...`, `hosted_vllm/...`). |
| `api_base`, `api_version`, `api_key` | `str` | `LLM`-specific. `api_base` is the LiteLLM-style endpoint override; both `base_url` and `api_base` are passed through, so either name works for the fallback path. |
| `reasoning_effort` | `"none" \| "low" \| "medium" \| "high"` | For o1/o3/o4 and compatible reasoning models. |
| `thinking` | `Any` | Anthropic extended thinking config. |
| `stream` | `bool` | Global default — `Crew(stream=True)` overrides per run. |
| `callbacks` | `list[Any]` | Passed to LiteLLM. |
| `timeout` | `float \| int \| None` | Per-call wall clock. |
| `context_window_size` | `int` | Override auto-detected context window. |

## Reasoning models

```python
from crewai import LLM

o3 = LLM(model="openai/o3", reasoning_effort="medium")

claude = LLM(
    model="anthropic/claude-sonnet-4-6",
    thinking={"type": "enabled", "budget_tokens": 4096},
)
```

- `reasoning_effort` maps onto OpenAI's `reasoning.effort`.
- `thinking` is passed through to Anthropic — the reasoning tokens show up in `CrewOutput.token_usage.reasoning_tokens` when enabled.
- GPT-5 and the o-series intentionally ignore `stop=` in 1.14+; the router strips it so you don't have to.

## Connecting to self-hosted / Ollama / vLLM

```python
LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434")
LLM(model="hosted_vllm/my-finetune", api_base="http://vllm:8000/v1", api_key="EMPTY")
```

Both of these take the **native OpenAI-compatible** path — no LiteLLM needed.

## Per-agent vs per-task LLMs

```python
from crewai import Agent, LLM

writer = Agent(
    role="Writer",
    goal="...",
    backstory="...",
    llm=LLM(model="openai/gpt-4o-mini"),
    function_calling_llm=LLM(model="openai/gpt-4o"),  # tool-call formatting on a smarter model
)
```

`function_calling_llm` is used **only** to format tool-call JSON arguments. Most models don't need it; leave it unset unless your primary model struggles with JSON schemas.

## Chat / manager / planner LLMs

`Crew` accepts three extra LLM slots:

| Field | Used by |
|---|---|
| `manager_llm` | Hierarchical process — spins up a default manager agent. |
| `planning_llm` | Crew-level `planning=True` planner. |
| `chat_llm` | `crewai chat` CLI against this crew. |

```python
from crewai import Crew, Process, LLM

Crew(
    agents=[...],
    tasks=[...],
    process=Process.hierarchical,
    manager_llm=LLM(model="openai/gpt-4o"),
    planning=True,
    planning_llm=LLM(model="openai/gpt-4o-mini"),
)
```

## Custom LLM — subclass `BaseLLM`

When you need a provider CrewAI doesn't support (or you want to proxy through your own gateway), subclass `BaseLLM`:

```python
from crewai.llms.base_llm import BaseLLM
from typing import Any

class MyGatewayLLM(BaseLLM):
    llm_type: str = "my_gateway"

    def call(self, messages: list[dict], **kwargs: Any) -> str:
        import httpx
        r = httpx.post(
            "https://gateway.internal/chat",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": self.temperature},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["text"]

agent = Agent(role="...", goal="...", backstory="...",
              llm=MyGatewayLLM(model="gw/llama-70b", api_key="..."))
```

Minimum surface: a `call()` method. Streaming and tool-use are opt-in — see the existing native providers under `crewai/llms/providers/` for reference implementations.

## Patterns

### 1. Environment-driven provider switch

```python
import os
from crewai import LLM

def default_llm():
    if os.getenv("USE_LOCAL"):
        return LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434")
    return LLM(model="openai/gpt-4o-mini")

agent = Agent(role="...", goal="...", backstory="...", llm=default_llm())
```

### 2. Cheap primary, smart function-calling

```python
Agent(
    role="Tool User",
    goal="...",
    backstory="...",
    llm=LLM(model="openai/gpt-4o-mini"),
    function_calling_llm=LLM(model="openai/gpt-4o"),
)
```

Only the tool-calling round uses the expensive model.

### 3. Observed latency/cost with callbacks

```python
class Timing:
    def log_pre_api_call(self, model, messages, kwargs):
        self.t0 = time.perf_counter()
    def log_post_api_call(self, *a, **k):
        print("call took", time.perf_counter() - self.t0, "s")

llm = LLM(model="openai/gpt-4o", callbacks=[Timing()])
```

Callbacks are passed through to LiteLLM — the standard LiteLLM callback API applies.

### 4. Structured output via provider-native mode

```python
from pydantic import BaseModel
class Plan(BaseModel):
    steps: list[str]

llm = LLM(model="openai/gpt-4o", response_format=Plan)
```

Works for OpenAI, Gemini 2, and Claude's structured-tool mode. For other providers set `Task.output_pydantic` instead.

### 5. LiteLLM routing config

```python
LLM(model="groq/llama3-70b", is_litellm=True, api_base=os.environ["LITELLM_PROXY"])
```

Forces LiteLLM so you benefit from its router, rate-limiting, and fallbacks.

## Gotchas

- **Provider prefix is checked against a whitelist.** Typos (`openaii/...`) fall through to LiteLLM, which then fails with a confusing error. Stick to the table above.
- **`is_anthropic` and `stop`** — the router strips `stop` for GPT-5 / o-series models in 1.14; passing it elsewhere works as expected.
- **LiteLLM is optional.** Without it installed, any non-native model raises `ImportError`. Install with `pip install litellm` or `uv add 'crewai[litellm]'`.
- **`max_tokens` vs `max_completion_tokens`.** OpenAI o-series wants the latter; LiteLLM maps both. If you set both, the native provider path picks `max_completion_tokens`.
- **Custom `BaseLLM` subclasses don't inherit streaming** — implement `call_with_streaming` / the streaming protocol yourself.
- **`reasoning=True` on Agent was renamed** — use `planning=True`. The LLM's `reasoning_effort` is separate and still valid.

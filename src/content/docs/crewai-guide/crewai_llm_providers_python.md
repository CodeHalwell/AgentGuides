---
title: "LLM providers and routing"
description: "How crewai.LLM picks a native SDK vs LiteLLM, the full list of supported providers, reasoning/thinking parameters, and building a custom BaseLLM."
framework: "crewai"
language: "python"
sidebar:
  label: "LLM providers"
  order: 25
---

> **Verified against `crewai` %%version=crewai%%** (checked %%versions-checked%%; source: `crewai/llm.py`, `crewai/llms/base_llm.py`, `crewai/llms/providers/`).

The `LLM` factory is doing more than it looks. It's a `__new__`-time router that picks between a native provider SDK and the LiteLLM fallback based on the model name — and it's stricter than older docs suggest.

## Minimal runnable example

```python
from crewai import LLM

# Native provider (OpenAI) — uses `openai` SDK directly
gpt = LLM(model="openai/%%model=openai-small%%", temperature=0.2)

# Provider prefix omitted — LLM infers from model name
gpt = LLM(model="%%model=openai-small%%")

# Non-native → LiteLLM fallback (needs `pip install "crewai[litellm]"`)
mistral = LLM(model="mistral/%%model=mistral-flagship%%")
```

`LLM(...)` returns an instance of the matching provider class (`OpenAICompletion` for the first two lines), not an `LLM` object. A bare model name is matched against known prefixes and name patterns (`gpt-*`, `o3-*`, `claude-*`, `gemini-*`); anything ambiguous should carry a provider prefix. Without LiteLLM installed, a model that is not served natively raises `ImportError` listing the native providers.

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
  │ 2. "/" in model ("openai/%%model=openai%%")   │
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

| Prefix | Canonical provider | SDK package (extra) |
|---|---|---|
| `openai` | openai | `openai` (installed by default) |
| `anthropic`, `claude` | anthropic | `anthropic` (`crewai[anthropic]`) |
| `azure`, `azure_openai` | azure | `azure-ai-inference` (`crewai[azure-ai-inference]`) |
| `google`, `gemini` | gemini | `google-genai` (`crewai[google-genai]`) |
| `bedrock`, `aws` | bedrock | `boto3` (`crewai[bedrock]`) |
| `snowflake` | snowflake | native Snowflake Cortex provider |
| `openrouter`, `deepseek`, `ollama`, `ollama_chat`, `hosted_vllm`, `cerebras`, `dashscope`, `deepinfra` | same | `openai`-compatible |

Anything else (`mistral/`, `groq/`, `cohere/`, custom prefixes) falls through to LiteLLM, which is an optional extra (`crewai[litellm]`).

## Force LiteLLM

```python
LLM(model="openai/%%model=openai-small%%", is_litellm=True)
```

Handy when you want LiteLLM's uniform interface even for natively supported models (e.g. to reuse a LiteLLM router config). It requires the `litellm` package; without it the call raises the same `ImportError` as an unsupported prefix.

## Constructor fields

What you can pass depends on which class `LLM(...)` returns. The common base is `BaseLLM`, whose fields are `model`, `temperature`, `top_p`, `max_tokens`, `stream`, `seed`, `frequency_penalty`, `presence_penalty`, `n`, `api_key`, `base_url`, `provider`, `stop` and `additional_params`. The native OpenAI class (`OpenAICompletion`) adds, among others, `timeout`, `max_retries` (default 2), `organization`, `project`, `default_headers`, `client_params`, `max_completion_tokens`, `response_format`, `logprobs`, `top_logprobs`, `reasoning_effort` and `api` (`"completions"` or `"responses"`, selecting the Responses API). The table below is the LiteLLM-backed `LLM` class, used for providers without a native integration or when `is_litellm=True`:

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
| `reasoning_effort` | `"none" \| "low" \| "medium" \| "high"` | For OpenAI reasoning models (GPT-5 and later, the o-series) and compatible providers. |
| `thinking` | `Any` | Anthropic extended thinking config. |
| `stream` | `bool` | Global default — `Crew(stream=True)` overrides per run. |
| `callbacks` | `list[Any]` | Passed to LiteLLM. |
| `timeout` | `float \| int \| None` | Per-call wall clock. |
| `context_window_size` | `int` | Override auto-detected context window. |

## Reasoning models

```python
from crewai import LLM

gpt = LLM(model="openai/%%model=openai-flagship%%", reasoning_effort="medium")

claude = LLM(
    model="anthropic/%%model=anthropic%%",
    thinking={"type": "enabled", "budget_tokens": 4096},
)
```

- `reasoning_effort` maps onto OpenAI's `reasoning.effort`.
- `thinking` is passed through to Anthropic — the reasoning tokens show up in `CrewOutput.token_usage.reasoning_tokens` when enabled.
- For GPT-5 and the o-series, `supports_stop_words()` returns `False`, so stop sequences are not sent; you can still pass `stop=` without an error.

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
    llm=LLM(model="openai/%%model=openai-small%%"),
    function_calling_llm=LLM(model="openai/%%model=openai%%"),  # tool-call formatting on a smarter model
)
```

`function_calling_llm` is used **only** to format tool-call JSON arguments. It is marked deprecated in the 1.15 line (on `Agent` and `Crew`); current models handle tool calls themselves, so leave it unset in new code. Each agent in a crew has its own `llm`, so "per-task LLM" means giving the task an agent with the model you want.

## Chat / manager / planner LLMs

`Crew` accepts three extra LLM slots:

| Field | Used by |
|---|---|
| `manager_llm` | Hierarchical process — spins up a default manager agent. |
| `planning_llm` | Crew-level `planning=True` planner. |
| `chat_llm` | `crewai chat` CLI against this crew. Also the analysis model for `Crew(memory=True)` when set (otherwise the first agent's LLM). |

```python
from crewai import Crew, Process, LLM

Crew(
    agents=[...],
    tasks=[...],
    process=Process.hierarchical,
    manager_llm=LLM(model="openai/%%model=openai%%"),
    planning=True,
    planning_llm=LLM(model="openai/%%model=openai-small%%"),
)
```

## Custom LLM — subclass `BaseLLM`

When you need a provider CrewAI doesn't support (or you want to proxy through your own gateway), subclass `BaseLLM`. `call()` is the only abstract method:

```python
from typing import Any

from crewai import Agent
from crewai.llms.base_llm import BaseLLM


class MyGatewayLLM(BaseLLM):
    llm_type: str = "my_gateway"

    def call(self, messages: str | list[dict], tools=None, callbacks=None,
             available_functions=None, from_task=None, from_agent=None,
             response_model=None, **kwargs: Any) -> str:
        import httpx

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        r = httpx.post(
            "https://gateway.internal/chat",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": self.temperature},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["text"]

    def get_context_window_size(self) -> int:
        return 128_000          # the default for an unknown model is 4096


agent = Agent(role="...", goal="...", backstory="...",
              llm=MyGatewayLLM(model="gw/llama-70b", api_key="..."))
```

What we checked by running a subclass like this with a patched HTTP client, in a crew and with `Agent.kickoff()`:

- The agent works with only `call()` implemented. The model must answer in CrewAI's text format (`Thought: ... Final Answer: ...`), because a subclass does not report native function calling by default: the agent takes the text (ReAct) path, and tools are described in the prompt.
- `messages` may be a plain string as well as a list of `{"role", "content"}` dicts; handle both, or accept `**kwargs` as above.
- `BaseLLM` fields are available as attributes (`self.model`, `self.api_key`, `self.temperature`, `self.stop`). A model name with a prefix (`gw/...`) sets `provider` to the prefix.
- The default context window for an unrecognised model is **4096 tokens**. With `respect_context_window=True` (the agent default) CrewAI summarises long conversations to fit, so override `get_context_window_size()` as above.
- Any subclass that defines `call()` or `acall()` is wrapped in CrewAI's rate-limit retry policy automatically.
- A subclass that only returns text emits no `LLMCall*` events and `CrewOutput.token_usage` stays at 0; the built-in providers emit the events and count tokens internally. Log usage yourself if you need it.
- For native tool calls, add `def supports_function_calling(self) -> bool: return True` and, when `tools` are passed, return a list of tool-call dicts such as `[{"id": "call_1", "function": {"name": "lookup_order", "arguments": "{\"order_id\": \"7\"}"}}]`; CrewAI executes them and calls the model again with the results. We ran this shape with a scripted model. The built-in providers under `crewai/llms/providers/` show how each vendor's response is converted.

## Patterns

### 1. Environment-driven provider switch

```python
import os
from crewai import LLM

def default_llm():
    if os.getenv("USE_LOCAL"):
        return LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434")
    return LLM(model="openai/%%model=openai-small%%")

agent = Agent(role="...", goal="...", backstory="...", llm=default_llm())
```

### 2. Cheap primary, smart function-calling

```python
Agent(
    role="Tool User",
    goal="Call the billing API correctly",
    backstory="Integration engineer.",
    llm=LLM(model="openai/%%model=openai-small%%"),
    function_calling_llm=LLM(model="openai/%%model=openai%%"),
)
```

Only the tool-calling round uses the expensive model. `function_calling_llm` is deprecated in 1.15, so treat this as a pattern for existing code rather than new designs.

### 3. Per-call latency and token usage

Native providers ignore `LLM(callbacks=[...])` (the source says "not used in native implementation"; they apply to the LiteLLM path only). For the built-in providers, listen for the LLM events instead:

```python
import time

from crewai.events import BaseEventListener, LLMCallCompletedEvent, LLMCallStartedEvent


class Timing(BaseEventListener):
    def __init__(self):
        self._started: dict[str, float] = {}
        super().__init__()

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(LLMCallStartedEvent)
        def started(source, event):
            self._started[event.call_id] = time.perf_counter()

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def completed(source, event):
            took = time.perf_counter() - self._started.pop(event.call_id)
            print(f"{event.model}: {took:.2f}s, usage={event.usage}")


timing = Timing()       # create it once, before any kickoff
```

We ran this against the native OpenAI provider pointed at a local stub server (`LLM(model="openai/...", base_url=...)`): it printed one line per call with the model name and a usage dict (`prompt_tokens`, `completion_tokens`, `total_tokens`).

### 4. Structured output via provider-native mode

```python
from pydantic import BaseModel
class Plan(BaseModel):
    steps: list[str]

llm = LLM(model="openai/%%model=openai%%", response_format=Plan)
```

Works where the provider integration supports structured output; for providers that do not, set `Task.output_pydantic`, which also parses the answer into `.pydantic`.

### 5. LiteLLM routing config

```python
import os
from crewai import LLM

LLM(model="groq/llama-3.3-70b-versatile", is_litellm=True, api_base=os.environ["LITELLM_PROXY"])
```

Forces LiteLLM so you benefit from its router, rate-limiting, and fallbacks.

## Gotchas

- **Provider prefix is checked against a list.** A typo such as `openaii/...` is not native, so CrewAI tries LiteLLM; without it installed you get an `ImportError` that lists the native providers. With LiteLLM installed the typo goes to LiteLLM and fails there. Stick to the table above.
- **`stop` and reasoning models.** GPT-5 and o-series models report `supports_stop_words() == False`; other OpenAI chat models apply `stop` normally.
- **LiteLLM is optional.** Without it installed, any non-native model raises `ImportError`. Install with `pip install litellm` or `uv add 'crewai[litellm]'`.
- **`max_tokens` vs `max_completion_tokens`.** OpenAI o-series wants the latter; LiteLLM maps both. If you set both, the native provider path picks `max_completion_tokens`.
- **Custom `BaseLLM` subclasses return whole responses.** There is no `call_with_streaming` hook to implement; `BaseLLM.stream_events()` wraps `call()` for streamed output, but chunk events come from the built-in providers.
- **`reasoning=True` on Agent was renamed** — use `planning=True`. The LLM's `reasoning_effort` is separate and still valid.

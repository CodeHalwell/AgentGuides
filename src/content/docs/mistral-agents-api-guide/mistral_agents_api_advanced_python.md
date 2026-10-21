---
title: "Mistral Agents API: Chat Completions and Advanced Python"
description: "Python patterns with the mistralai SDK outside the Agents API: tool calling on chat completions, structured outputs, async fan-out, per-call retries and streaming."
framework: "mistral-agents-api"
language: "python"
sidebar:
  label: "Chat completions and advanced Python"
  order: 5
---

Verified against `mistralai` %%version=mistralai%% (checked %%versions-checked%%). Upstream: [client-python](https://github.com/mistralai/client-python) · [PyPI](https://pypi.org/project/mistralai/) · [docs](https://docs.mistral.ai).

Not every agent needs the stateful Agents API. When you hold the conversation yourself (for example, inside an existing chat backend), `chat.complete()` with function tools is the lighter option. Since SDK v3 the built-in tools (`web_search`, `code_interpreter` and friends) are not accepted there; use [Conversations](/mistral-agents-api-guide/mistral_agents_api_comprehensive_guide/) for those.

## Patterns

- Tool reliability: validate arguments, time-box tool execution, return errors to the model as results.
- Structured outputs validated with Pydantic.
- Concurrency bounded with semaphores; retries configured per call where it matters.

## Tool calling with chat completions

```python
import json
import os
from mistralai.client import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": 18, "condition": "cloudy"}   # call your real API here

messages: list = [{"role": "user", "content": "What's the weather in Paris and in Lyon?"}]

for _ in range(5):                                    # bound the loop
    response = client.chat.complete(model="mistral-medium-latest", messages=messages, tools=tools)
    message = response.choices[0].message
    if not message.tool_calls:
        print(message.content)
        break
    messages.append(message)
    for call in message.tool_calls:
        args = json.loads(call.function.arguments) if isinstance(call.function.arguments, str) else call.function.arguments
        try:
            result = get_weather(**args)
        except Exception as exc:                      # let the model see the failure
            result = {"error": str(exc)}
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.function.name,
            "content": json.dumps(result),
        })
```

`parallel_tool_calls=False` forces one call at a time; `tool_choice="any"` forces a tool call.

## Structured outputs

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    supplier: str
    total_eur: float
    due_date: str

parsed = client.chat.parse(
    response_format=Invoice,
    model="mistral-small-latest",
    messages=[{"role": "user", "content": "Extract: ACME SARL invoice, 1,240.50 EUR due 2026-11-30."}],
)
invoice = parsed.choices[0].message.parsed
print(invoice.total_eur)
```

For agents, pass `response_format_from_pydantic_model(Invoice)` in `completion_args`, or use `RunContext(output_format=Invoice)`.

## Async fan-out with bounded concurrency

```python
import asyncio
from mistralai.client import Mistral

async def summarise_all(texts: list[str], concurrency: int = 8) -> list[str]:
    limit = asyncio.Semaphore(concurrency)
    async with Mistral() as client:
        async def one(text: str) -> str:
            async with limit:
                r = await client.chat.complete_async(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": f"Summarise in one sentence:\n\n{text}"}],
                )
                return r.choices[0].message.content
        return await asyncio.gather(*(one(t) for t in texts))
```

For thousands of independent requests, the Batch API (`client.batch.jobs.create(...)`) is cheaper and avoids rate limits.

## Retries and timeouts per call

```python
from mistralai.client.utils import BackoffStrategy, RetryConfig

patient = RetryConfig("backoff", BackoffStrategy(1000, 30_000, 2.0, 180_000), retry_connection_errors=True)

response = client.chat.complete(
    model="mistral-medium-latest",
    messages=[{"role": "user", "content": "Draft a release note for v4.2."}],
    retries=patient,
    timeout_ms=120_000,
)
```

## Streaming

```python
with client.chat.stream(
    model="mistral-medium-latest",
    messages=[{"role": "user", "content": "Explain idempotency keys."}],
) as stream:
    for event in stream:
        delta = event.data.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
```

## Observability and deployment

The SDK's OpenTelemetry spans are opt-in; set-up (`configure_telemetry`, redaction, Mistral's own traces and evaluations) is in the [production guide](/mistral-agents-api-guide/mistral_agents_api_production_guide/#3-observability), together with the [Dockerfile, Kubernetes manifest and CI workflow](/mistral-agents-api-guide/mistral_agents_api_production_guide/#10-deployment-and-cicd). Nothing there is specific to the Agents API, so it applies unchanged to a service built on `chat.complete()`.

## Security

- Keep keys in a secret manager; prefer service-account tokens that rotate.
- Separate read and write tools, and use least-privileged credentials for each.
- Allow-list network egress for tool code.
- Avoid putting prompts or personal data in traces and logs.

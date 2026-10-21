---
title: "Mistral Agents API Advanced (Python)"
description: "Advanced Python patterns with the mistralai SDK: tool calling on chat completions, structured outputs, async fan-out, per-call retries, tracing and deployment."
framework: mistral-agents-api
---

# Mistral Agents API Advanced (Python)

Verified against `mistralai` %%version=mistralai%% (checked %%versions-checked%%). Upstream: [client-python](https://github.com/mistralai/client-python) · [PyPI](https://pypi.org/project/mistralai/) · [docs](https://docs.mistral.ai).

Not every agent needs the stateful Agents API. When you hold the conversation yourself (for example, inside an existing chat backend), `chat.complete()` with function tools is the lighter option. Since SDK v3 the built-in tools (`web_search`, `code_interpreter` and friends) are not accepted there; use [Conversations](/mistral-agents-api-guide/mistral_agents_api_comprehensive_guide/) for those.

## Patterns

- Tool reliability: validate arguments, time-box tool execution, return errors to the model as results.
- Structured outputs validated with Pydantic.
- Concurrency bounded with semaphores; retries configured per call where it matters.
- Tracing with OpenTelemetry.

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

## Observability

The SDK depends on `opentelemetry-api` and records spans for its HTTP calls when your process has a tracer provider configured. Install `mistralai[telemetry]` for the OTLP exporter:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces")))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("weather-agent")

with tracer.start_as_current_span("answer_weather_question") as span:
    response = client.chat.complete(model="mistral-small-latest", messages=[{"role": "user", "content": "Hi"}])
    span.set_attribute("llm.tokens.total", response.usage.total_tokens)
```

Record per request: model, tokens, tool names, tool durations and the final outcome.

## Deployment (Kubernetes)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mistral-agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mistral-agent
  template:
    metadata:
      labels:
        app: mistral-agent
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/mistral-agent:1.0.0
          env:
            - name: MISTRAL_API_KEY
              valueFrom:
                secretKeyRef:
                  name: mistral-secrets
                  key: apiKey
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet: { path: /healthz, port: 8080 }
---
apiVersion: v1
kind: Service
metadata:
  name: mistral-agent
spec:
  selector:
    app: mistral-agent
  ports:
    - port: 80
      targetPort: 8080
```

To use a projected service-account token instead of an API key, mount it as a file and set `MISTRAL_SA_TOKEN_PATH` to its path; the SDK re-reads it on every request.

## CI/CD (GitHub Actions)

```yaml
name: deploy
on:
  push:
    branches: [main]
jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
```

## Security

- Keep keys in a secret manager; prefer service-account tokens that rotate.
- Separate read and write tools, and use least-privileged credentials for each.
- Allow-list network egress for tool code.
- Avoid putting prompts or personal data in traces and logs.

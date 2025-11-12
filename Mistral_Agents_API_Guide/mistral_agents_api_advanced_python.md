---
layout: default
title: "Mistral Agents API Advanced (Python)"
description: "Advanced patterns for Mistral agents: error handling, observability, deployment."
---

# Mistral Agents API Advanced (Python)


Latest: 1.9.11
Upstream: https://github.com/mistralai/client-python/releases | https://docs.mistral.ai/getting-started/clients | https://pypi.org/project/mistralai/

## Patterns
- Tool reliability with retries/backoff
- Structured outputs and validation
- Tracing and metrics with OTEL

## Tool-Calling Chat (Python SDK)

```python
from mistralai import Mistral
import os, json

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather by city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

messages = [
    {"role": "user", "content": "What's the weather in Paris?"}
]

resp = client.chat.complete(
    model="mistral-large-latest",
    messages=messages,
    tools=tools,
)

msg = resp.choices[0].message
if msg.tool_calls:
    for call in msg.tool_calls:
        if call.function.name == "get_weather":
            args = json.loads(call.function.arguments)
            # call your real weather API here
            tool_result = {"temp_c": 18, "condition": "cloudy", "city": args["city"]}
            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": json.dumps(tool_result),
            })
            final = client.chat.complete(model="mistral-large-latest", messages=messages)
            print(final.choices[0].message.content)
```

## Resilience
- Wrap downstream API calls (weather, DB) with retries/backoff and timeouts
- Validate tool outputs (Pydantic) before sending back to model

## Observability
- Trace each step: user -> tool call -> tool result -> final answer
- Record tokens, model, tool names, and durations

## Streaming

```python
for event in client.chat.stream(model="mistral-large-latest", messages=messages, tools=tools):
    # event could be tool_call, content_delta, or completed
    print(event)
```

## Deployment (Kubernetes)

deployment.yaml

{% raw %}
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
          image: ghcr.io/yourorg/mistral-agent:latest
          env:
            - name: MISTRAL_API_KEY
              valueFrom:
                secretKeyRef:
                  name: mistral-secrets
                  key: apiKey
          ports:
            - containerPort: 8080
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

## CI/CD (GitHub Actions)

.github/workflows/deploy.yml

```yaml
{% raw %}
name: deploy
on:
  push:
    branches: [ main ]
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
          tags: ghcr.io/${{ github.repository }}:latest
  deploy:
    needs: docker
    runs-on: ubuntu-latest
    steps:
      - uses: azure/k8s-deploy@v5
        with:
          action: deploy
          manifests: |
            deployment.yaml
          images: ghcr.io/${{ github.repository }}:latest
          namespace: default
{% endraw %}
```
{% endraw %}

## Security Best Practices
- Store keys in cloud secret managers (Key Vault/Secret Manager/Secrets Manager)
- Use least-privileged IAM roles; separate read/write roles for tools
- Network egress allowlisting for tool calls
- Encrypt logs; avoid sensitive data in traces
- Rotate keys regularly; monitor for anomalies

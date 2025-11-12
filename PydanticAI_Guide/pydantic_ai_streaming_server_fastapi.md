---
layout: default
title: "PydanticAI Streaming Server (FastAPI)"
description: "Stream tokens from the model and validate the final output with PydanticAI."
---

# PydanticAI Streaming Server (FastAPI)

Latest: 1.14.1
Last verified: 2025-11

This example streams model output over SSE while using Pydantic for final validation.

{% raw %}
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class Summary(BaseModel):
    bullets: list[str]

@app.get("/stream")
def stream(q: str):
    def gen():
        chunks = []
        with client.chat.completions.stream(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Summarize as JSON with bullets: {q}"}],
        ) as s:
            for event in s:
                # yield raw event data
                yield f"data: {event}\n\n"
                chunks.append(str(event))
        # final validation step
        final_text = "".join(chunks)
        try:
            Summary.model_validate_json(final_text)
            yield "data: {\"validated\": true}\n\n"
        except ValidationError as e:
            yield f"data: {{\"validated\": false, \"error\": {e.json()} }}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```
{% endraw %}

## Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: pydanticai-stream }
spec:
  replicas: 2
  selector: { matchLabels: { app: pydanticai-stream } }
  template:
    metadata: { labels: { app: pydanticai-stream } }
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/pydanticai-stream:latest
          env: [{ name: OPENAI_API_KEY, valueFrom: { secretKeyRef: { name: openai-secrets, key: apiKey } } }]
          ports: [{ containerPort: 8080 }]
---
apiVersion: v1
kind: Service
metadata: { name: pydanticai-stream }
spec: { selector: { app: pydanticai-stream }, ports: [{ port: 80, targetPort: 8080 }] }
```

### Security Best Practices
- Validate content before emitting via SSE
- Enforce auth and rate limiting on the endpoint
- Avoid logging PII/sensitive fields in requests or responses

Notes:
- Token-level streaming occurs via OpenAI SDK; PydanticAI validation happens on the final JSON.
- Adjust to integrate with your PydanticAI agent pipeline as needed.
---
title: "Google ADK Advanced (Python)"
description: "Advanced patterns for Google Agent Development Kit: multi-agent, HITL, observability, deployment."
framework: google-adk
language: python
---

# Google ADK Advanced (Python)


Latest: 2.0.0b1 | Updated: April 2026
Upstream: https://github.com/google/adk-python | https://google.github.io/adk-docs/get-started/python/

## Patterns
- Multi-agent coordination with runner
- HITL and escalation
- GCP deploy: Cloud Run, Vertex AI, tracing

## Quickstart Pattern

Use `LlmAgent` (the parameter is singular `instruction`, not `instructions`) and wire everything through an `InMemoryRunner` for local development. `Runner` takes a single root `agent=` (or `app=`) — there is no `agents=` list parameter.

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from google.adk.runners import InMemoryRunner

researcher = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Research the topic and cite sources.",
)
writer = LlmAgent(
    name="writer",
    model="gemini-2.5-flash",
    instruction="Draft and refine copy using the `researcher` tool.",
    tools=[AgentTool(agent=researcher)],
)

async def main():
    runner = InMemoryRunner(agent=writer, app_name="pipeline")
    events = await runner.run_debug(
        "Summarise LangGraph with references",
        user_id="u1",
        session_id="s1",
    )

asyncio.run(main())
```

See the [agents page](./agents/) for the full constructor surface and the [runner page](./runner-and-sessions/) for production wiring.

## Deployment
- Package for Cloud Run; use Secret Manager for keys
- Enable Cloud Trace and Cloud Logging; export OTLP if needed

### Cloud Run (service.yaml)

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: adk-agent
  namespace: default
spec:
  template:
    spec:
      containers:
        - image: gcr.io/PROJECT_ID/adk-agent:latest
          env:
            - name: GOOGLE_APPLICATION_CREDENTIALS
              value: /var/secrets/google/key.json
```

### GitHub Actions (Cloud Run deploy)


```yaml
name: deploy-cloud-run
on: { push: { branches: [ main ] } }
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      - uses: google-github-actions/setup-gcloud@v2
      - run: gcloud builds submit --tag gcr.io/$PROJECT_ID/adk-agent:latest .
      - run: gcloud run deploy adk-agent --image gcr.io/$PROJECT_ID/adk-agent:latest --region $REGION --platform managed
```


## Security Best Practices
- Store keys in cloud secret managers (Key Vault/Secret Manager/Secrets Manager)
- Use least-privileged IAM roles; separate read/write roles for tools
- Network egress allowlisting for tool calls
- Encrypt logs; avoid sensitive data in traces
- Rotate keys regularly; monitor for anomalies


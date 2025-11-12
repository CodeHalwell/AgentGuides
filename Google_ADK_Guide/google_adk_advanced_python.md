---
layout: default
title: "Google ADK Advanced (Python)"
description: "Advanced patterns for Google Agent Development Kit: multi-agent, HITL, observability, deployment."
---

# Google ADK Advanced (Python)


Latest: 1.18.0
Upstream: https://github.com/google/adk-python | https://google.github.io/adk-docs/get-started/python/

## Patterns
- Multi-agent coordination with runner
- HITL and escalation
- GCP deploy: Cloud Run, Vertex AI, tracing

## Quickstart Pattern

```python
from google.adk import Agent, Runner

researcher = Agent(name="researcher", instructions="Research and cite sources.")
writer = Agent(name="writer", instructions="Draft and refine copy.")

runner = Runner(agents=[researcher, writer])
result = runner.run("Summarize LangGraph with references")
print(result.output)
```

## Deployment
- Package for Cloud Run; use Secret Manager for keys
- Enable Cloud Trace and Cloud Logging; export OTLP if needed

### Cloud Run (service.yaml)

{% raw %}
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
{% endraw %}

## Security Best Practices
- Store keys in cloud secret managers (Key Vault/Secret Manager/Secrets Manager)
- Use least-privileged IAM roles; separate read/write roles for tools
- Network egress allowlisting for tool calls
- Encrypt logs; avoid sensitive data in traces
- Rotate keys regularly; monitor for anomalies


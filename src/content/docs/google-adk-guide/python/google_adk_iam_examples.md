---
title: "Google ADK - IAM Examples"
description: "Least-privilege IAM roles for ADK on GCP."
framework: google-adk
language: python
---

# Google ADK - IAM Examples

## Roles
- Vertex AI: roles/aiplatform.user
- Cloud Run: roles/run.invoker (runtime), roles/run.admin (deploy)
- Secret Manager: roles/secretmanager.secretAccessor
- Logging/Monitoring: roles/logging.logWriter, roles/monitoring.metricWriter

## gcloud Example

```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:adk-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```


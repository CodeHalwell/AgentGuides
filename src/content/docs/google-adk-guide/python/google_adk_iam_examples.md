---
title: "Google ADK — IAM, Auth & Credentials"
description: "Least-privilege IAM roles, service account setup, AuthenticatedFunctionTool, Secret Manager integration, and credential patterns for google-adk 2.1.0."
framework: google-adk
language: python
sidebar:
  order: 80
---

Verified against google-adk==2.3.0 (`google/adk/auth/`, `google/adk/integrations/secret_manager/`, `google/adk/tools/authenticated_function_tool.py`).

This guide covers the full credential story for ADK agents deployed on Google Cloud: which IAM roles each service needs, how to create and bind a service account, how to inject secrets via Secret Manager, and how to wire OAuth2 / API-key auth into your tools with `AuthenticatedFunctionTool`.

---

## 1 · Minimum IAM roles by service

Grant these roles to the service account that **runs** your ADK application.

| Scenario | Role | Purpose |
|---|---|---|
| Call Gemini models | `roles/aiplatform.user` | Vertex AI inference |
| Read/write Vertex AI Agent Engine sessions | `roles/aiplatform.user` | VertexAiSessionService |
| Use Vertex AI Memory Bank | `roles/aiplatform.user` | VertexAiMemoryBankService |
| Read GCS artifacts | `roles/storage.objectViewer` | GcsArtifactService loads |
| Write GCS artifacts | `roles/storage.objectAdmin` | GcsArtifactService saves |
| Read Secret Manager secrets | `roles/secretmanager.secretAccessor` | API keys, OAuth secrets |
| Write logs | `roles/logging.logWriter` | Structured logging |
| Write metrics | `roles/monitoring.metricWriter` | Cloud Monitoring |
| Deploy to Cloud Run | `roles/run.admin` | `gcloud run deploy` |
| Invoke a Cloud Run service | `roles/run.invoker` | Inter-service calls |
| Publish to Pub/Sub | `roles/pubsub.publisher` | PubSubToolset publish |
| Subscribe from Pub/Sub | `roles/pubsub.subscriber` | PubSubToolset pull |
| Query BigQuery | `roles/bigquery.dataViewer` + `roles/bigquery.jobUser` | BigQuery tools |
| Spanner read | `roles/spanner.databaseReader` | SpannerToolset |
| Spanner read/write | `roles/spanner.databaseUser` | SpannerToolset mutations |

---

## 2 · Creating and binding a service account

### Create the SA and grant roles

```bash
PROJECT_ID="my-gcp-project"
SA_NAME="adk-runner"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# 1. Create the service account
gcloud iam service-accounts create "${SA_NAME}" \
  --project="${PROJECT_ID}" \
  --display-name="ADK Runtime Service Account"

# 2. Grant Vertex AI access (Gemini, Agent Engine)
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user"

# 3. Grant GCS artifact storage
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin" \
  --condition="expression=resource.name.startsWith('projects/_/buckets/my-adk-artifacts'),title=adk-bucket-only"

# 4. Grant Secret Manager access
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"

# 5. Grant logging
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/logging.logWriter"
```

### Deploy to Cloud Run with the SA

```bash
# Build and deploy — the SA runs the container, Workload Identity handles model calls
gcloud run deploy adk-agent \
  --image="gcr.io/${PROJECT_ID}/adk-agent:latest" \
  --region="us-central1" \
  --service-account="${SA_EMAIL}" \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=us-central1" \
  --no-allow-unauthenticated
```

### Key JSON file (local dev only)

```bash
# Create a key for local development — do NOT commit this file
gcloud iam service-accounts keys create sa-key.json \
  --iam-account="${SA_EMAIL}"

# Tell ADK / google-auth to use it
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/sa-key.json"
```

On Cloud Run / GKE / Compute Engine, omit `GOOGLE_APPLICATION_CREDENTIALS` entirely — Application Default Credentials (ADC) use the attached SA automatically.

---

## 3 · Workload Identity Federation (GKE)

On GKE, use Workload Identity so your pods inherit IAM roles without a JSON key:

```bash
# 1. Enable Workload Identity on the cluster (once)
gcloud container clusters update my-cluster \
  --workload-pool="${PROJECT_ID}.svc.id.goog"

# 2. Create a Kubernetes service account
kubectl create serviceaccount adk-ksa --namespace default

# 3. Bind the KSA → GSA
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[default/adk-ksa]"

# 4. Annotate the KSA
kubectl annotate serviceaccount adk-ksa \
  --namespace default \
  iam.gke.io/gcp-service-account="${SA_EMAIL}"
```

In your Deployment manifest, reference `serviceAccountName: adk-ksa`. ADC within the pod resolves automatically.

---

## 4 · `ServiceAccount` credential in code

For tools that call GCP APIs directly (e.g. an OpenAPI toolset that wraps a private GCP endpoint), pass credentials programmatically:

```python
from google.adk.auth.auth_credential import (
    AuthCredential,
    AuthCredentialTypes,
    ServiceAccount,
    ServiceAccountCredential,
)
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import CustomAuthScheme

# ── From a JSON key file loaded at startup ────────────────────────────────────
import json, pathlib

sa_json = json.loads(pathlib.Path("sa-key.json").read_text())

sa_cred = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=ServiceAccount(
        service_account_credential=ServiceAccountCredential(
            type_="service_account",
            project_id=sa_json["project_id"],
            private_key_id=sa_json["private_key_id"],
            private_key=sa_json["private_key"],
            client_email=sa_json["client_email"],
            client_id=sa_json["client_id"],
            auth_uri=sa_json["auth_uri"],
            token_uri=sa_json["token_uri"],
            auth_provider_x509_cert_url=sa_json["auth_provider_x509_cert_url"],
            client_x509_cert_url=sa_json["client_x509_cert_url"],
            universe_domain=sa_json.get("universe_domain", "googleapis.com"),
        ),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    ),
)

# ── Using ADC (no key file) ───────────────────────────────────────────────────
adc_cred = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=ServiceAccount(
        use_default_credential=True,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    ),
)
```

---

## 5 · `AuthenticatedFunctionTool` — automatic credential injection

`AuthenticatedFunctionTool` (experimental) is a `FunctionTool` subclass that runs the ADK auth flow before your function is called. It handles:

1. **First call** — credential not yet obtained → requests it from the client (OAuth redirect, API-key prompt, etc.) and returns `response_for_auth_required`.
2. **Subsequent call** — credential obtained → injects it as the `credential` kwarg and calls your function.

### API-key tool

```python
import httpx
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes
from google.adk.agents import LlmAgent
from fastapi.security import APIKeyHeader

# Declare the auth scheme (OpenAPI-style)
from google.adk.auth.auth_schemes import CustomAuthScheme

api_key_scheme = CustomAuthScheme(type="apiKey", **{"in": "header", "name": "X-API-Key"})

# Pre-configured credential (you already have the key)
api_key_cred = AuthCredential(
    auth_type=AuthCredentialTypes.API_KEY,
    api_key="sk-my-secret-api-key",
)

auth_cfg = AuthConfig(
    auth_scheme=api_key_scheme,
    raw_auth_credential=api_key_cred,
)

async def call_weather_api(city: str, credential) -> dict:
    """Fetch current weather for a city.

    Args:
      city: City name, e.g. 'London'.
    Returns:
      A dict with temperature and description.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://weather.example.com/v1/current",
            params={"city": city},
            headers={"X-API-Key": credential.api_key},
        )
        resp.raise_for_status()
        return resp.json()

weather_tool = AuthenticatedFunctionTool(
    func=call_weather_api,
    auth_config=auth_cfg,
)

agent = LlmAgent(
    name="weather_bot",
    model="gemini-2.5-flash",
    instruction="Answer weather questions using the call_weather_api tool.",
    tools=[weather_tool],
)
```

### OAuth2 Authorization Code flow (3-legged)

Use this when the user must grant consent (e.g. Google Calendar, Salesforce):

```python
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool

# Auth scheme — discovery via well-known OIDC endpoint
google_oidc = OpenIdConnectWithConfig(
    authorization_endpoint="https://accounts.google.com/o/oauth2/auth",
    token_endpoint="https://oauth2.googleapis.com/token",
    scopes=["https://www.googleapis.com/auth/calendar.readonly"],
)

# Client credential — only client_id/client_secret, no access_token yet
client_cred = AuthCredential(
    auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
    oauth2=OAuth2Auth(
        client_id="YOUR_CLIENT_ID.apps.googleusercontent.com",
        client_secret="YOUR_CLIENT_SECRET",
        redirect_uri="https://myapp.example.com/oauth/callback",
    ),
)

calendar_auth = AuthConfig(
    auth_scheme=google_oidc,
    raw_auth_credential=client_cred,
    credential_key="google-calendar",   # reuse across tool calls
)

async def list_calendar_events(max_results: int = 10, credential=None) -> dict:
    """List upcoming calendar events.

    Args:
      max_results: Maximum number of events to return.
    Returns:
      A dict with 'events' list.
    """
    import httpx
    access_token = credential.oauth2.access_token if credential else None
    if not access_token:
        return {"error": "no credential"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            params={"maxResults": max_results, "orderBy": "startTime", "singleEvents": True},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        return resp.json()

calendar_tool = AuthenticatedFunctionTool(
    func=list_calendar_events,
    auth_config=calendar_auth,
    response_for_auth_required={
        "status": "auth_required",
        "message": "Please authorise Google Calendar access.",
    },
)
```

**Flow at runtime:**
1. User asks "What's on my calendar tomorrow?"
2. First call: `credential` not yet exchanged → returns `{"status": "auth_required", "message": "..."}` and sets `actions.requested_auth_configs`.
3. The ADK web client (or your server) shows the OAuth redirect to the user.
4. User grants consent → the runner calls the tool again with the exchanged token.
5. `call_calendar_events` runs normally with `credential.oauth2.access_token` populated.

### Client Credentials flow (service-to-service)

For machine-to-machine calls where no user consent is needed:

```python
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.auth.auth_schemes import CustomAuthScheme
from google.adk.auth.auth_tool import AuthConfig
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool

m2m_scheme = CustomAuthScheme(type="oauth2")  # custom, service-to-service

m2m_cred = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id="svc-client-id",
        client_secret="svc-client-secret",
        auth_uri="https://auth.example.com/oauth/authorize",
    ),
)

m2m_auth = AuthConfig(
    auth_scheme=m2m_scheme,
    raw_auth_credential=m2m_cred,
    credential_key="internal-m2m",
)

async def submit_order(order_json: str, credential=None) -> dict:
    """Submit an order to the internal order service.

    Args:
      order_json: JSON string with order details.
    Returns:
      A dict with order_id.
    """
    import json, httpx
    token = credential.oauth2.access_token if credential else ""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://orders.internal/v2/orders",
            json=json.loads(order_json),
            headers={"Authorization": f"Bearer {token}"},
        )
        return resp.json()

order_tool = AuthenticatedFunctionTool(func=submit_order, auth_config=m2m_auth)
```

### Bearer token (already obtained)

When you already have an access token (e.g. forwarded from the calling user):

```python
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, HttpAuth, HttpCredentials
from google.adk.auth.auth_tool import AuthConfig
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool

bearer_cred = AuthCredential(
    auth_type=AuthCredentialTypes.HTTP,
    http=HttpAuth(
        scheme="bearer",
        credentials=HttpCredentials(token="ya29.ALREADY_OBTAINED_TOKEN"),
    ),
)

bearer_auth = AuthConfig(
    auth_scheme=CustomAuthScheme(type="http", scheme="bearer"),
    raw_auth_credential=bearer_cred,
)
```

---

## 6 · Secret Manager integration

ADK ships a `SecretManagerClient` in `google.adk.integrations.secret_manager` to load secrets at startup without embedding them in code.

```python
from google.adk.integrations.secret_manager.secret_client import SecretManagerClient

# --- Using Application Default Credentials (Cloud Run, GKE) ------------------
client = SecretManagerClient()   # falls back to ADC when no key/token given

# --- Using a service account JSON key (local dev) ----------------------------
import pathlib
client = SecretManagerClient(
    service_account_json=pathlib.Path("sa-key.json").read_text(),
)

# --- Access a secret version --------------------------------------------------
secret_value = client.get_secret(
    project_id="my-gcp-project",
    secret_id="openai-api-key",
    version="latest",          # or "3" for a specific version
)
```

**Load secrets at agent startup:**

```python
import os
from google.adk.integrations.secret_manager.secret_client import SecretManagerClient
from google.adk.agents import LlmAgent
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool

def build_agent() -> LlmAgent:
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    client = SecretManagerClient()

    # Pull the API key at import time — not baked into the image
    stripe_key = client.get_secret(project, "stripe-api-key", "latest")
    db_password = client.get_secret(project, "pg-password", "latest")

    stripe_auth = AuthConfig(
        auth_scheme=CustomAuthScheme(type="apiKey", **{"in": "header", "name": "Stripe-API-Key"}),
        raw_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.API_KEY,
            api_key=stripe_key,
        ),
    )

    async def charge_card(amount_cents: int, token: str, credential=None) -> dict:
        """Charge a payment card via Stripe.

        Args:
          amount_cents: Amount to charge in cents.
          token: Stripe card token.
        Returns:
          A dict with charge_id.
        """
        import httpx
        async with httpx.AsyncClient() as c:
            resp = await c.post(
                "https://api.stripe.com/v1/charges",
                data={"amount": amount_cents, "currency": "usd", "source": token},
                auth=(credential.api_key if credential else "", ""),
            )
            return resp.json()

    payment_tool = AuthenticatedFunctionTool(func=charge_card, auth_config=stripe_auth)
    return LlmAgent(
        name="payment_agent",
        model="gemini-2.5-flash",
        instruction="Process payment requests using charge_card.",
        tools=[payment_tool],
    )

root_agent = build_agent()
```

**Required IAM for Secret Manager:**

```bash
gcloud secrets add-iam-policy-binding "stripe-api-key" \
  --project="${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"
```

---

## 7 · Manual credential request from a plain `FunctionTool`

When you want the OAuth flow without `AuthenticatedFunctionTool`, use `ToolContext` directly:

```python
from google.adk.tools.tool_context import ToolContext
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth

DRIVE_AUTH = AuthConfig(
    auth_scheme=OpenIdConnectWithConfig(
        authorization_endpoint="https://accounts.google.com/o/oauth2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    ),
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id="CLIENT_ID.apps.googleusercontent.com",
            client_secret="CLIENT_SECRET",
        ),
    ),
)

async def list_drive_files(folder_id: str, tool_context: ToolContext) -> dict:
    """List files in a Google Drive folder.

    Args:
      folder_id: The Google Drive folder ID.
    Returns:
      A dict with 'files' list or 'status': 'auth_required'.
    """
    cred = tool_context.get_auth_response(DRIVE_AUTH)
    if cred is None:
        # Pause and send the OAuth redirect to the UI
        tool_context.request_credential(DRIVE_AUTH)
        return {"status": "auth_required"}

    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://www.googleapis.com/drive/v3/files",
            params={"q": f"'{folder_id}' in parents"},
            headers={"Authorization": f"Bearer {cred.oauth2.access_token}"},
        )
        return resp.json()
```

---

## 8 · Patterns

### 1 — Per-tenant API keys in session state

Store each tenant's API key in session state under `user:` prefix (persisted across sessions):

```python
from google.adk.tools.tool_context import ToolContext

async def call_tenant_api(endpoint: str, tool_context: ToolContext) -> dict:
    """Call a tenant-specific API endpoint."""
    api_key = tool_context.state.get("user:api_key")
    if not api_key:
        return {"error": "No API key configured for this user."}
    import httpx
    async with httpx.AsyncClient() as c:
        resp = await c.get(endpoint, headers={"Authorization": f"Bearer {api_key}"})
        return {"status": resp.status_code, "body": resp.json()}
```

Set the key via state delta at session creation:

```python
session = await runner.session_service.create_session(
    app_name="demo",
    user_id="tenant-42",
    state={"user:api_key": "sk-tenant-42-secret"},
)
```

### 2 — Workload Identity + Impersonation

For agents that call other Google APIs on behalf of users:

```bash
# Grant the ADK SA permission to impersonate a higher-privileged SA
gcloud iam service-accounts add-iam-policy-binding "privileged-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/iam.serviceAccountTokenCreator"
```

Then use `ServiceAccount(use_default_credential=True)` in code — ADC resolves the attached SA and the token creator binding handles elevation.

### 3 — Rotate secrets without redeployment

Store the API key version in Secret Manager. On each new deploy, `SecretManagerClient.get_secret(..., version="latest")` always fetches the current live secret. Rotate by publishing a new version in Secret Manager — no code change needed.

### 4 — Cross-project Secret Manager access

Grant `roles/secretmanager.secretAccessor` on the **secret** (not just the project) for fine-grained control:

```bash
gcloud secrets add-iam-policy-binding "my-secret" \
  --project="other-project" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"
```

---

## 9 · Gotchas

- `AuthenticatedFunctionTool` is `@experimental(FeatureName.PLUGGABLE_AUTH)` — suppress the warning with `GOOGLE_ADK_IGNORE_WARNINGS=pluggable_auth`.
- `credential_key` in `AuthConfig` allows two tools to share a cached token. Without it, each tool triggers its own auth flow even when they use the same provider.
- `AuthenticatedFunctionTool` marks `"credential"` as an `_ignore_params` name — it does **not** appear in the model's function schema. Always name the parameter exactly `credential` in your function signature.
- On Cloud Run with min-instances=0, the SA key rotation is picked up on cold start. Warm instances keep the old secret until their memory is cleared.
- `SecretManagerClient` with no arguments tries ADC first, then raises `ValueError` if ADC is not configured. Always set `GOOGLE_APPLICATION_CREDENTIALS` in local dev environments.
- The `roles/aiplatform.user` role is required for **every** Vertex AI API call — including Agent Engine sessions and memory. If you see `403 PERMISSION_DENIED` from Vertex, this role is missing.
- Never log `credential.oauth2.access_token` or `credential.api_key` — redact these in all callback and plugin logging.

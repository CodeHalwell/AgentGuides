---
title: "Google ADK Production Guide (Python)"
description: "Deploying Google ADK agents with the adk CLI and the Agent Platform SDK: Agent Engine, Cloud Run and GKE, plus persistent sessions, authentication, telemetry, cost controls, guardrails and testing."
framework: "google-adk"
language: "python"
sidebar:
  label: "Production"
  order: 90
---

Checked against `google-adk` %%version=google-adk%% on %%versions-checked%%. The `adk deploy` options below are taken from the CLI's own `--help` and source, the Python examples were run locally against a stub model, and the Agent Platform SDK calls were checked against `google-cloud-aiplatform` 2.6. Nothing here was deployed to a live project, so treat the cloud commands as a starting point and read the output of the first run.

This page covers what ADK itself gives you for production. For IAM roles and credentials see [IAM, auth and credentials](/google-adk-guide/python/google_adk_iam_examples/); for the command line as a whole see [Project layout and the `adk` CLI](/google-adk-guide/python/google_adk_comprehensive_guide/#project-layout-and-the-adk-cli).

## Deployment targets

ADK deploys an **agent folder**: a Python package whose `agent.py` defines `root_agent` (an agent, or a `Workflow`), or that holds a `root_agent.yaml`. Every target runs the same FastAPI server as `adk api_server`.

| Target | Command | What you get |
|---|---|---|
| Vertex AI Agent Engine | `adk deploy agent_engine` | Managed runtime with Agent Platform Sessions and Memory Bank wired in, scaling and IAM handled by Google Cloud |
| Cloud Run | `adk deploy cloud_run` | A container built by Cloud Build from source (`gcloud run deploy --source`), stored in Artifact Registry |
| GKE | `adk deploy gke` | A container built by Cloud Build, plus a Deployment and a `ClusterIP` Service applied with `kubectl` |
| Local Docker | `adk deploy docker` | The same image, run locally, for testing the container |
| Your own server | `get_fast_api_app(...)` | The ADK app as a FastAPI object you can add routes and middleware to, then deploy however you like |

Pick Agent Engine when you want managed sessions and memory and no container to run. Pick Cloud Run or GKE when the agent has to sit next to your other services, in your VPC, behind your own authentication.

## Prepare the agent folder

```text
my_agent/
├── __init__.py          # from . import agent
├── agent.py             # defines root_agent
├── requirements.txt     # your dependencies; google-adk is added by the deployer
├── .env                 # local development only (listed in .gitignore)
└── .gitignore
```

`adk create my_agent` produces this layout. Two things to know before the first deploy:

- **Ignored files stay behind.** `adk deploy` copies the folder but skips anything matched by its `.gitignore`, `.gcloudignore` or `.ae_ignore`, and always skips the local `.adk/` folder of development sessions.
- **`.env` is read by `adk deploy agent_engine`.** Its values (apart from the project, region and service account, which become deploy settings) are forwarded to the deployed agent as plain environment variables. Keep API keys and passwords out of `.env` for deployment and use Secret Manager instead (see [Secrets](#secrets) below).

Pin the ADK version you tested with. Each deploy command takes `--adk_version`, which defaults to the version installed where you run it.

## Agent Engine with `adk deploy`

```bash
gcloud services enable aiplatform.googleapis.com cloudbuild.googleapis.com

adk deploy agent_engine \
  --project "$PROJECT_ID" \
  --region europe-west2 \
  --display_name "support-agent" \
  --service_account "support-agent@$PROJECT_ID.iam.gserviceaccount.com" \
  --otel_to_cloud \
  my_agent
```

What this does (from `cli/cli_deploy.py`):

1. Stages the folder, adds `google-cloud-aiplatform[adk,agent_engines]` and `google-adk` to `requirements.txt`, and writes a Dockerfile that runs `adk api_server`.
2. Creates an Agent Engine instance (a `reasoningEngines` resource) and uploads the source; Cloud Build builds the image.
3. Points sessions and memory at the instance itself (`agentengine://<resource>`), so conversations and memories persist without extra configuration.
4. Prints the resource name. Pass it back as `--agent_engine_id` to update the same instance instead of creating a new one.

Useful options:

| Option | Purpose |
|---|---|
| `--service_account` | Runtime identity (new in 2.12.0). Without it Agent Engine uses its default service agent. Give the account only the roles the agent needs. |
| `--agent_engine_id` | Update an existing instance in place. |
| `--otel_to_cloud` | Export traces, metrics and logs to Cloud Observability. Prompt and response text is not captured in spans unless you set `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS`. |
| `--worker_pool` | Build with a private Cloud Build pool (VPC Service Controls, private networks). |
| `--extra_packages` | Ship local packages that live outside the agent folder. |
| `--trigger_sources pubsub,eventarc` | Register `/apps/{app}/trigger/*` endpoints for event-driven runs; restrict callers with `--trigger_oidc_audience` and `--trigger_oidc_service_accounts`. |
| `--agent_engine_config_file` | Read settings such as scaling and labels from `.agent_engine_config.json`; command-line flags override it. |

`--staging_bucket`, `--adk_app`, `--env_file`, `--requirements_file` and `--trace_to_cloud` are accepted but deprecated and ignored or due for removal.

## Agent Engine from Python

When the deployment is part of a larger script or pipeline, use the Agent Platform SDK (the `agentplatform` package in `google-cloud-aiplatform` 2.x; the older `vertexai.Client` still works but warns that it is deprecated). `AdkApp` wraps an agent or an `App` and is also how you test the deployed behaviour locally:

```python
import asyncio

from google.adk.agents import Agent
from agentplatform.frameworks import AdkApp

root_agent = Agent(
    name="support",
    model="%%model=gemini%%",
    instruction="Answer briefly and politely.",
)

app = AdkApp(agent=root_agent)


async def main():
    # Locally, AdkApp uses in-memory sessions; on Agent Engine it uses Agent Platform Sessions.
    session = await app.async_create_session(user_id="u1")
    async for event in app.async_stream_query(
        user_id="u1", session_id=session["id"], message="Where is my order?"
    ):
        print(event["content"]["parts"][0]["text"])


asyncio.run(main())
```

Events come back as dictionaries, not `Event` objects, because they are the same JSON the remote service returns. To deploy the same object:

```python
import agentplatform

client = agentplatform.Client(project="my-project", location="europe-west2")

remote = client.runtimes.create(
    agent=app,                                   # the AdkApp from above
    config={
        "display_name": "support-agent",
        "staging_bucket": "gs://my-project-agent-staging",   # required when passing an object
        "requirements": ["google-cloud-aiplatform[adk,agent_engines]>=2.2,<3"],
        "service_account": "support-agent@my-project.iam.gserviceaccount.com",
        "env_vars": {
            "SUPPORT_API_KEY": {"secret": "support-api-key", "version": "latest"},  # Secret Manager
        },
        "min_instances": 1,
    },
)
print(remote.api_resource.name)   # projects/.../locations/.../reasoningEngines/...
```

`client.runtimes.get(name=...)` returns a handle with `async_stream_query`, `async_create_session` and the other `AdkApp` methods, now running remotely. `client.runtimes.update(...)` and `client.runtimes.delete(...)` manage the instance. A value in `env_vars` that is a dictionary becomes a Secret Manager reference rather than a plain variable.

## Cloud Run

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

adk deploy cloud_run \
  --project "$PROJECT_ID" \
  --region europe-west2 \
  --service_name support-agent \
  --session_service_uri "postgresql+asyncpg://agent@/sessions?host=/cloudsql/$PROJECT_ID:europe-west2:agents" \
  --artifact_service_uri "gs://$PROJECT_ID-agent-artifacts" \
  --otel_to_cloud \
  my_agent \
  -- --no-allow-unauthenticated \
     --service-account "support-agent@$PROJECT_ID.iam.gserviceaccount.com" \
     --add-cloudsql-instances "$PROJECT_ID:europe-west2:agents" \
     --set-secrets "DB_PASSWORD=agent-db-password:latest" \
     --min-instances 1
```

- Everything after `--` goes to `gcloud run deploy`. ADK does not choose an authentication mode for you; without `--no-allow-unauthenticated` (or `--allow-unauthenticated`) gcloud asks interactively. Keep the service private unless it really is public.
- The image is built from source by Cloud Build and stored in Artifact Registry (the `cloud-run-source-deploy` repository in the region).
- **Choose session storage.** With no `--session_service_uri`, Cloud Run gets in-memory sessions (the container's folder is not writable), so conversations are lost when an instance stops and are not shared between instances. Use a database URI (`postgresql+asyncpg://…`, `mysql+aiomysql://…`) or `agentengine://<resource>` to use Agent Platform Sessions. Artifacts default to in-memory too; `gs://bucket` stores them in Cloud Storage.
- `--with_ui` also deploys the Dev UI. It is a development tool; leave it off in production.
- `--a2a` adds the A2A endpoint; see [MCP and A2A](/google-adk-guide/python/mcp-and-a2a/).

Callers need `roles/run.invoker` and an identity token:

```bash
gcloud run services add-iam-policy-binding support-agent --region europe-west2 \
  --member "serviceAccount:frontend@$PROJECT_ID.iam.gserviceaccount.com" --role roles/run.invoker

URL=$(gcloud run services describe support-agent --region europe-west2 --format 'value(status.url)')
TOKEN=$(gcloud auth print-identity-token)
curl -s -X POST "$URL/apps/my_agent/users/u1/sessions" -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{}'
```

The runtime service account needs `roles/aiplatform.user` for Gemini on Vertex AI, plus whatever the session and artifact stores need (`roles/cloudsql.client`, `roles/storage.objectUser`, `roles/secretmanager.secretAccessor`).

## GKE

```bash
adk deploy gke \
  --project "$PROJECT_ID" \
  --region europe-west2 \
  --cluster_name agents \
  --service_name support-agent \
  --session_service_uri "postgresql+asyncpg://..." \
  --otel_to_cloud \
  my_agent
```

`adk deploy gke` builds the image with Cloud Build, fetches cluster credentials, and applies a Deployment and Service. The Service is `ClusterIP` by default, so the agent is reachable only inside the cluster; `--service_type LoadBalancer` exposes a public IP, with no authentication in front of it, so put an authenticating gateway or Identity-Aware Proxy in front instead.

The command tags the image as `gcr.io/PROJECT/SERVICE:TIMESTAMP`. Since Container Registry was shut down, `gcr.io` paths only work in projects that have `gcr.io` repositories in Artifact Registry. If yours does not, or if you need your own manifests (Workload Identity, resource limits, a `PodDisruptionBudget`), build the image yourself and push it to an Artifact Registry repository (`REGION-docker.pkg.dev/PROJECT/REPO/IMAGE`). The image ADK generates is simple: a `python:3.11-slim` base, your folder under `/app/agents`, `pip install google-adk` plus your `requirements.txt`, a non-root user, and `adk api_server` with the service URIs as its command. The generated files are deleted after each deploy.

## Your own server

`adk api_server` and every deploy target serve `get_fast_api_app()`. Call it yourself when you need extra routes or middleware, such as your own authentication:

```python
# main.py, next to the my_agent/ folder
import os

from google.adk.cli.fast_api import get_fast_api_app

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))

app = get_fast_api_app(
    agents_dir=AGENTS_DIR,
    session_service_uri=os.environ.get("SESSION_SERVICE_URI", "sqlite:///./sessions.db"),
    allow_origins=["https://app.example.com"],
    web=False,   # API only; the Dev UI is for local development
)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}
```

Run it with `uvicorn main:app --host 0.0.0.0 --port 8080`. The main endpoints are `POST /apps/{app}/users/{user}/sessions` to create a session, `POST /run` for a complete turn and `POST /run_sse` to stream events (the run is cancelled if the client disconnects). The server's endpoints are unauthenticated: put it behind Cloud Run IAM, IAP or your own middleware. Other useful arguments: `memory_service_uri`, `artifact_service_uri`, `otel_to_cloud`, `a2a`, `max_llm_calls` and `auto_create_session`.

## Sessions, memory and artifacts

The `--*_service_uri` options (and the matching `get_fast_api_app` arguments) choose the storage:

| Service | URI | Notes |
|---|---|---|
| Sessions | `postgresql+asyncpg://…`, `mysql+aiomysql://…`, `sqlite:///path.db` | `DatabaseSessionService`; needs `google-adk[db]` and the driver |
| Sessions | `agentengine://<resource or id>` | Agent Platform Sessions; the default on Agent Engine |
| Memory | `agentengine://<resource or id>` | Memory Bank; the default on Agent Engine |
| Memory | `rag://<corpus>` | Vertex AI RAG memory |
| Artifacts | `gs://bucket` | `GcsArtifactService` |
| Any | `memory://` | In-memory, for tests |

In code, the same services are constructor arguments on `Runner` (see [Runner, App and Sessions](/google-adk-guide/python/runner-and-sessions/)).

## Secrets

- **Cloud Run:** `--set-secrets NAME=secret:version` after `--`; the value arrives as an environment variable or a mounted file.
- **Agent Engine from Python:** a dictionary value in `env_vars`, as above.
- **Inside tools:** ADK's Secret Manager client (`google.adk.integrations.secret_manager`) and authenticated tools are covered in [IAM, auth and credentials](/google-adk-guide/python/google_adk_iam_examples/).

Never put keys in `instruction` text or session state: both are sent to the model and stored with the session.

## Observability

- `--otel_to_cloud` (all deploy targets, `adk web`, `adk api_server`) exports OpenTelemetry traces, metrics and logs to Cloud Trace, Cloud Monitoring and Cloud Logging. Install `google-adk[gcp]` for the exporters when you run your own server.
- ADK emits spans for each invocation, agent run, model call and tool call, and `gen_ai.client.*` metrics for token usage and latency.
- In your own process, `google.adk.telemetry.setup.maybe_set_otel_providers()` with exporters from `google.adk.telemetry.google_cloud.get_gcp_exporters(enable_cloud_tracing=True, ...)` sets the same thing up by hand.

For your own usage log, a runner plugin sees every model call across all agents:

```python
import logging

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin

log = logging.getLogger("agent.usage")


class UsagePlugin(BasePlugin):
    """Logs one structured line per completed model call."""

    def __init__(self) -> None:
        super().__init__(name="usage")

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        usage = llm_response.usage_metadata
        if usage and not llm_response.partial:
            log.info(
                "model_call",
                extra={
                    "agent": callback_context.agent_name,
                    "invocation_id": callback_context.invocation_id,
                    "prompt_tokens": usage.prompt_token_count,
                    "output_tokens": usage.candidates_token_count,
                    "cached_tokens": usage.cached_content_token_count,
                },
            )
        return None   # leave the response unchanged


root_agent = Agent(name="support", model="%%model=gemini%%", instruction="Answer briefly.")
app = App(name="support", root_agent=root_agent, plugins=[UsagePlugin()])
```

`BigQueryAgentAnalyticsPlugin` (`google-adk[bigquery-analytics]`) writes the full event stream to BigQuery instead; see [Callbacks and plugins](/google-adk-guide/python/callbacks-and-plugins/).

## Cost and context controls

These live on the `App` and `RunConfig`, not in your infrastructure:

```python
from google.adk.agents import Agent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.agents.run_config import RunConfig
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig

root_agent = Agent(
    name="support",
    model="%%model=gemini%%",
    static_instruction="You are the support agent for Example Ltd. ...",  # stable prefix, cache-friendly
    instruction="The customer's tier is {user:tier?}.",
)

app = App(
    name="support",
    root_agent=root_agent,
    # Summarise older events so long sessions do not resend the whole history.
    events_compaction_config=EventsCompactionConfig(compaction_interval=10, overlap_size=2),
    # Explicit Gemini context caching for the stable part of the prompt.
    context_cache_config=ContextCacheConfig(ttl_seconds=1800, min_tokens=4096),
)

run_config = RunConfig(max_llm_calls=20)   # hard cap on model calls per invocation
```

- **Compaction** (`EventsCompactionConfig`) replaces older events with an LLM-written summary every `compaction_interval` invocations, or once the prompt passes `token_threshold`.
- **Context caching** (`ContextCacheConfig`) creates Gemini cached content for the stable prefix. Gemini applies its own minimum size, so short prompts are not cached; check `cached_content_token_count` in the usage log above. Implicit caching works without configuration if stable text comes first, which is what `static_instruction` is for.
- **`max_llm_calls`** stops runaway tool loops. `adk api_server` and `get_fast_api_app` also accept `max_llm_calls`, and the `ADK_MAX_LLM_CALLS` environment variable sets a process-wide default.
- **Model choice per agent.** Give routers and classifiers a small model (`%%model=gemini-small%%`) and keep the larger one for the agents that need it.

Details and the exact trigger rules are in [Runner, App and Sessions](/google-adk-guide/python/runner-and-sessions/#context-caching-contextcacheconfig-experimental).

## Reliability

```python
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import FallbackModel
from google.adk.models.google_llm import Gemini
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.genai import types

primary = Gemini(
    model="%%model=gemini%%",
    retry_options=types.HttpRetryOptions(attempts=4, initial_delay=1.0, max_delay=20.0),
)

root_agent = Agent(
    name="support",
    # Try the primary model (with its retries), then the backup on 429/5xx.
    model=FallbackModel(models=[primary, "%%model=gemini-small%%"]),
    instruction="Answer briefly.",
)

app = App(
    name="support",
    root_agent=root_agent,
    # When a tool raises, show the error to the model and let it correct its call.
    plugins=[ReflectAndRetryToolPlugin(max_retries=2)],
)
```

- `Gemini(retry_options=...)` retries one model on transient HTTP errors.
- `FallbackModel` (2.9.0, experimental) moves to the next model only on 429 and 5xx, and only before the first response chunk, so a streamed turn is never stitched together from two models.
- `ReflectAndRetryToolPlugin` handles tool failures; `on_tool_error_callback` gives you full control.
- For long, multi-step runs, `App(resumability_config=ResumabilityConfig(is_resumable=True))` lets an interrupted invocation resume from where it stopped. Workflow nodes that fail are re-run on resume, so make their side effects idempotent.

## Guardrails

The Model Armor plugin (2.8.0, `google-adk[gcp]`) screens user input, model output and tool output with Model Armor templates you create in the console or with `gcloud model-armor templates create`:

```python
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.integrations.model_armor import ModelArmorConfig, ModelArmorPlugin

TEMPLATE = "projects/my-project/locations/europe-west2/templates/agent-default"

root_agent = Agent(name="support", model="%%model=gemini%%", instruction="Answer briefly.")

app = App(
    name="support",
    root_agent=root_agent,
    plugins=[
        ModelArmorPlugin(
            config=ModelArmorConfig(
                prompt_template_name=TEMPLATE,
                response_template_name=TEMPLATE,
                tool_output_template_name=TEMPLATE,
                deidentify_sensitive_data=True,   # pass on de-identified text when only SDP matched
            )
        )
    ],
)
```

Blocked content is replaced with the configured message, and by default a screening failure blocks too (`block_on_screening_failure=True`). For rules of your own, a plugin's `before_tool_callback` can refuse a call by returning a result dictionary, and `after_tool_callback` can filter what a tool returns before the model sees it (2.12.0). Gemini's own `safety_settings` go in the agent's `generate_content_config`.

## Testing and CI

- **Unit tests without a model.** Register a `BaseLlm` subclass for test model names, or pass a stub model instance to the agent, and drive it with `InMemoryRunner(...).run_debug(...)`.
- **Behaviour tests.** `adk eval my_agent my_agent/tests/smoke.evalset.json --config_file_path my_agent/test_config.json` runs eval sets and prints a pass/fail summary; `AgentEvaluator` does the same from pytest. See [Evaluation](/google-adk-guide/python/evaluation/).
- **Deploy from CI** by running the same `adk deploy …` command in Cloud Build or GitHub Actions, authenticated with Workload Identity Federation rather than a key file. Pass `--agent_engine_id` (Agent Engine) or a fixed `--service_name` (Cloud Run, GKE) so each run updates the existing deployment.

A minimal Cloud Build step:

```yaml
steps:
  - name: python:3.13-slim
    entrypoint: bash
    args:
      - -c
      - |
        pip install "google-adk[eval]==%%version=google-adk%%" pytest
        adk eval my_agent my_agent/tests/smoke.evalset.json --config_file_path my_agent/test_config.json
  - name: gcr.io/google.com/cloudsdktool/google-cloud-cli:slim
    entrypoint: bash
    args:
      - -c
      - |
        pip install "google-adk==%%version=google-adk%%"
        adk deploy cloud_run --project "$PROJECT_ID" --region europe-west2 \
          --service_name support-agent my_agent -- --no-allow-unauthenticated
```

Check that the build image has the Python version ADK needs (3.11 or later) before relying on the second step.

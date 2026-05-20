---
title: "Google ADK Advanced (Python)"
description: "Planners (BuiltInPlanner, PlanReActPlanner), custom BaseTool / BaseToolset, authenticated tools, and deployment."
framework: google-adk
language: python
sidebar:
  order: 80
---

Verified against google-adk==2.0.0 (`google/adk/planners/`, `google/adk/tools/`, `google/adk/auth/`).

## Planners

Planners modify how an `LlmAgent` reasons before it acts. Two implementations ship with ADK; you can also subclass `BasePlanner`.

### `BuiltInPlanner` — native model thinking

`BuiltInPlanner` wraps Gemini's built-in chain-of-thought thinking feature. It injects `thinking_config` into every `LlmRequest` emitted by the agent. No extra prompt engineering needed — the model reasons internally.

```python
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

# Thinking with a token budget
agent = LlmAgent(
    name="analyst",
    model="gemini-2.5-pro",           # thinking is a Gemini 2.5 feature
    instruction="Analyse the dataset step-by-step and produce a concise summary.",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,      # thoughts appear as Part(thought=True) in events
            thinking_budget=8192,       # max tokens dedicated to reasoning
        )
    ),
)
```

When `include_thoughts=True` is set, events will contain `Part` objects where `part.thought is True`. Filter them out when rendering to users:

```python
async for event in runner.run_async(...):
    if event.content:
        visible_text = "".join(
            p.text or ""
            for p in event.content.parts or []
            if p.text and not p.thought
        )
        if visible_text:
            print(visible_text)
```

`BuiltInPlanner` source (`planners/built_in_planner.py`):

- `build_planning_instruction` returns `None` — no extra system prompt is injected.
- `process_planning_response` returns `None` — response parts are not post-processed.
- `apply_thinking_config` is the actual workhorse — it sets `llm_request.config.thinking_config`.

**Precedence:** If you also set `thinking_config` inside `generate_content_config` on the agent, the planner's config wins (a `UserWarning` is emitted).

### `PlanReActPlanner` — textual Plan-then-Act

`PlanReActPlanner` does not require model-native thinking. Instead, it appends a system instruction that asks the model to output its reasoning wrapped in special tags (`/*PLANNING*/`, `/*REASONING*/`, `/*ACTION*/`, `/*FINAL_ANSWER*/`). The planner's `process_planning_response` strips these into `thought=True` parts before returning.

```python
from google.adk.planners import PlanReActPlanner
from google.adk.agents import LlmAgent

agent = LlmAgent(
    name="planner_bot",
    model="gemini-2.5-flash",          # works with any model — no native thinking needed
    instruction="You are a research assistant.",
    tools=[my_search_tool, my_db_tool],
    planner=PlanReActPlanner(),
)
```

The model will produce text like:

```
/*PLANNING*/
I need to search for X, then look up Y in the database.
/*PLANNING*/

/*ACTION*/
[function call: search(query="X")]
```

ADK classifies `/*PLANNING*/` / `/*REASONING*/` / `/*ACTION*/` blocks as thoughts and `/*FINAL_ANSWER*/` as the user-visible response.

When to choose `PlanReActPlanner` over `BuiltInPlanner`:
- You are using a non-Gemini model that doesn't support built-in thinking.
- You want the plan to be **visible in the conversation history** (it becomes a `thought=True` part).
- You want model-independent, reproducible reasoning traces.

### Custom `BasePlanner`

Implement `build_planning_instruction` to inject custom system text and `process_planning_response` to post-process parts.

```python
from typing import List, Optional
from google.genai import types
from google.adk.planners.base_planner import BasePlanner
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest


class BudgetPlanner(BasePlanner):
    """Inject a token-budget hint into the system instruction."""

    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps

    def build_planning_instruction(
        self,
        readonly_context: ReadonlyContext,
        llm_request: LlmRequest,
    ) -> Optional[str]:
        return (
            f"You have at most {self.max_steps} tool calls to answer. "
            "Plan carefully before acting."
        )

    def process_planning_response(
        self,
        callback_context: CallbackContext,
        response_parts: List[types.Part],
    ) -> Optional[List[types.Part]]:
        return None   # no post-processing needed


agent = LlmAgent(
    name="budget_agent",
    model="gemini-2.5-flash",
    planner=BudgetPlanner(max_steps=3),
    tools=[...],
)
```

## Custom `BaseTool`

`FunctionTool` covers most use cases. Subclass `BaseTool` when you need full control over the `FunctionDeclaration` sent to the model or when the tool must mutate the `LlmRequest` before it leaves (e.g. inject inline data, set response_modalities).

```python
from typing import Any, Optional
from google.genai import types
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext


class DatabaseLookupTool(BaseTool):
    """Queries an internal product database by SKU."""

    def __init__(self, db_pool):
        super().__init__(
            name="lookup_product",
            description="Retrieve product details by SKU from the internal database.",
        )
        self._db = db_pool

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "sku": types.Schema(
                        type=types.Type.STRING,
                        description="The SKU identifier (e.g. 'WIDGET-42').",
                    ),
                },
                required=["sku"],
            ),
        )

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> dict:
        sku = args.get("sku")
        if not sku:
            return {"error": "sku is required"}
        row = await self._db.fetchrow(
            "SELECT name, price, stock FROM products WHERE sku = $1", sku
        )
        if row is None:
            return {"error": f"SKU {sku!r} not found"}
        tool_context.state["last_sku"] = sku
        return {"name": row["name"], "price": row["price"], "stock": row["stock"]}


# Usage
db_tool = DatabaseLookupTool(db_pool=my_pool)
agent = LlmAgent(
    name="catalog",
    model="gemini-2.5-flash",
    instruction="Use lookup_product to answer product questions.",
    tools=[db_tool],
)
```

### Overriding `process_llm_request`

Use this when you need to inject tool-related data into the **request** before it reaches the model — not the same as a before_model callback:

```python
from google.adk.models.llm_request import LlmRequest

class ComputerUseTool(BaseTool):
    """Injects a computer-use tool declaration into the LlmRequest."""

    def __init__(self):
        super().__init__(
            name="computer_use",
            description="Control the browser/desktop.",
        )

    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        return None   # built-in; not declared in config.tools

    async def process_llm_request(
        self, *, tool_context: ToolContext, llm_request: LlmRequest
    ) -> None:
        # Inject a built-in computer-use block
        llm_request.config = llm_request.config or types.GenerateContentConfig()
        llm_request.config.tools = llm_request.config.tools or []
        llm_request.config.tools.append(
            types.Tool(computer_use=types.ToolComputerUse(...))
        )

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> dict:
        ...
```

## Custom `BaseToolset`

`BaseToolset` exposes a dynamic list of tools per invocation — useful when the available tools vary by user, tenant, or session state.

```python
from typing import Optional
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.readonly_context import ReadonlyContext


class TenantToolset(BaseToolset):
    """Exposes only the tools the current tenant has permission to use."""

    def __init__(self, permission_service):
        super().__init__()
        self._permissions = permission_service

    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> list[BaseTool]:
        if readonly_context is None:
            return []

        tenant_id = readonly_context.state.get("tenant_id")
        allowed = await self._permissions.get_allowed_tools(tenant_id)

        tools = []
        if "search" in allowed:
            tools.append(FunctionTool(func=self._search))
        if "write" in allowed:
            tools.append(FunctionTool(func=self._write_record))
        return tools

    async def _search(self, query: str) -> dict:
        """Search tenant data.

        Args:
          query: The search query.
        Returns:
          A dict with key `results`.
        """
        ...

    async def _write_record(self, record: dict) -> dict:
        """Write a record.

        Args:
          record: The record to write.
        Returns:
          A dict with key `id` of the created record.
        """
        ...

    async def close(self) -> None:
        # release any held connections
        await self._permissions.close()


# Usage — toolset is resolved fresh on every invocation
agent = LlmAgent(
    name="tenant_agent",
    model="gemini-2.5-flash",
    tools=[TenantToolset(permission_service=svc)],
)
```

### Built-in toolset caching

`BaseToolset.get_tools_with_prefix` (called by the framework) caches the tool list per invocation ID. If your toolset state doesn't change within a turn, this is transparent. To **disable** the cache (for toolsets that change tool availability mid-turn):

```python
class VolatileToolset(BaseToolset):
    def __init__(self):
        super().__init__()
        self._use_invocation_cache = False   # force re-eval on every LLM call

    async def get_tools(self, readonly_context=None) -> list[BaseTool]:
        ...
```

### Tool name prefixing

`tool_name_prefix` prevents collisions when the same toolset is registered multiple times (e.g. per-tenant filesystem toolsets):

```python
fs_a = TenantToolset(permission_service=svc_a)
fs_a.tool_name_prefix = "tenant_a"

fs_b = TenantToolset(permission_service=svc_b)
fs_b.tool_name_prefix = "tenant_b"

# Model sees: tenant_a_search, tenant_b_search, etc.
agent = LlmAgent(name="root", tools=[fs_a, fs_b])
```

## Authenticated tools

### API-key tools

```python
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import APIKeyScheme
from google.adk.auth.auth_credential import AuthCredential, APIKey

async def call_external_api(endpoint: str, tool_context: ToolContext) -> dict:
    """Call the external analytics API.

    Args:
      endpoint: The API endpoint path.
    Returns:
      A dict with the API response data.
    """
    cred = tool_context.get_auth_response(
        AuthConfig(auth_scheme=APIKeyScheme(name="X-Api-Key"))
    )
    api_key = cred.api_key.value if cred else None
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://analytics.example.com{endpoint}",
            headers={"X-Api-Key": api_key},
        )
        return resp.json()

tool = FunctionTool(func=call_external_api)
```

### OAuth2 / OIDC tools

```python
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.auth_credential import AuthCredential, OAuth2Auth
from google.adk.tools.tool_context import ToolContext

GITHUB_OIDC = OpenIdConnectWithConfig(
    authorization_endpoint="https://github.com/login/oauth/authorize",
    token_endpoint="https://github.com/login/oauth/access_token",
    scopes=["read:user", "repo"],
)

GITHUB_AUTH_CONFIG = AuthConfig(
    auth_scheme=GITHUB_OIDC,
    raw_auth_credential=AuthCredential(
        auth_type="oauth2",
        oauth2=OAuth2Auth(
            client_id="YOUR_CLIENT_ID",
            client_secret="YOUR_CLIENT_SECRET",
        ),
    ),
)

async def list_github_repos(tool_context: ToolContext) -> dict:
    """List the authenticated user's GitHub repositories.

    Returns:
      A dict with key `repos` containing a list of repo names.
    """
    cred = tool_context.get_auth_response(GITHUB_AUTH_CONFIG)
    if cred is None:
        # First call — ADK will trigger the OAuth flow
        tool_context.request_credential(GITHUB_AUTH_CONFIG)
        return {"status": "auth_required"}
    token = cred.oauth2.access_token
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.github.com/user/repos",
            headers={"Authorization": f"Bearer {token}"},
        )
        repos = [r["name"] for r in resp.json()]
    return {"repos": repos}
```

## Multi-agent coordination

### Fan-out with `AgentTool`

Calling multiple specialists concurrently from a coordinator:

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from google.adk.runners import InMemoryRunner
from pydantic import BaseModel

class ResearchQuery(BaseModel):
    query: str

class CodeQuery(BaseModel):
    question: str

researcher = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Research the query and cite sources.",
    input_schema=ResearchQuery,
    tools=[google_search],
)

coder = LlmAgent(
    name="coder",
    model="gemini-2.5-pro",
    instruction="Answer the coding question with a minimal Python example.",
    input_schema=CodeQuery,
)

coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction=(
        "For factual questions use `researcher`. "
        "For coding questions use `coder`. "
        "For questions mixing both, call both in parallel then synthesise."
    ),
    tools=[
        AgentTool(agent=researcher),
        AgentTool(agent=coder),
    ],
)
```

### Chaining via `output_key` and `{state}` templates

```python
drafter = LlmAgent(
    name="drafter",
    model="gemini-2.5-flash",
    instruction="Write a 3-sentence press release about {topic}.",
    output_key="draft",           # writes the reply to session.state["draft"]
)

editor = LlmAgent(
    name="editor",
    model="gemini-2.5-pro",
    instruction="Polish the following draft:\n\n{draft}\n\nReturn only the final text.",
    include_contents="none",      # ignore history; read only from state
)
```

Run both in a `Workflow` sequential pipeline; `{draft}` in the editor's instruction resolves to `session.state["draft"]` at runtime.

## Human-in-the-loop (HITL)

### Long-running tool

```python
from google.adk.tools import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
import asyncio

async def request_approval(
    amount: float,
    description: str,
    tool_context: ToolContext,
) -> dict:
    """Request manager approval for an expense.

    Args:
      amount: Expense amount in USD.
      description: What the expense is for.
    Returns:
      A dict with `status` ('pending', 'approved', or 'rejected').
    """
    ticket_id = f"EXP-{int(asyncio.get_event_loop().time())}"
    tool_context.state["approval_ticket"] = ticket_id
    # Fire-and-forget: notify the approver via Slack / email
    await notify_approver(ticket_id, amount, description)
    return {
        "status": "pending",
        "ticket_id": ticket_id,
        "message": "Approval request sent. The manager will respond within 24 hours.",
    }

approval_tool = LongRunningFunctionTool(func=request_approval)

agent = LlmAgent(
    name="expense_bot",
    model="gemini-2.5-flash",
    instruction="Help the user submit expense reports.",
    tools=[approval_tool],
)
```

When the manager responds (via webhook or another invocation), call `runner.run_async` again with a `FunctionResponse` carrying the decision:

```python
from google.genai import types

await runner.run_async(
    user_id="u1",
    session_id="s1",
    new_message=types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name="request_approval",
                    id=tool_context.function_call_id,
                    response={"status": "approved"},
                )
            )
        ],
    ),
)
```

## Deployment — Cloud Run (FastAPI)

```python
# main.py — FastAPI wrapper for Cloud Run
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.plugins import LoggingPlugin
from google.genai import types

SESSION_DB = os.environ["SESSION_DB_URL"]

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

app_container = App(
    name="assistant_app",
    root_agent=agent,
    plugins=[LoggingPlugin()],
)

runner: Runner | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner
    runner = Runner(
        app=app_container,
        session_service=DatabaseSessionService(db_url=SESSION_DB),
    )
    yield
    await runner.close()


web_app = FastAPI(lifespan=lifespan)


@web_app.post("/chat/{session_id}")
async def chat(session_id: str, body: dict):
    user_id = body.get("user_id", "anonymous")
    text = body.get("message", "")
    if not text:
        raise HTTPException(status_code=400, detail="message is required")

    # Ensure session exists
    svc = runner.session_service
    if not await svc.get_session(app_name="assistant_app", user_id=user_id, session_id=session_id):
        await svc.create_session(
            app_name="assistant_app", user_id=user_id, session_id=session_id
        )

    response_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    ):
        if event.is_final_response() and event.content:
            response_text = "".join(p.text or "" for p in event.content.parts or [])
    return {"reply": response_text}
```

**Cloud Run Dockerfile:**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "main:web_app", "--host", "0.0.0.0", "--port", "8080"]
```

**Deploy:**

```bash
gcloud run deploy adk-assistant \
  --source . \
  --region us-central1 \
  --set-env-vars SESSION_DB_URL=postgresql+asyncpg://user:pass@host/db \
  --set-secrets GOOGLE_API_KEY=adk-api-key:latest \
  --allow-unauthenticated
```

## Observability — OpenTelemetry

ADK emits OpenTelemetry traces via `google.adk.telemetry`. Export to Cloud Trace:

```python
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(CloudTraceSpanExporter())
)
trace.set_tracer_provider(provider)
```

Or export to any OTLP-compatible backend (Jaeger, Tempo, etc.):

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317"))
)
```

Every `run_async` call creates a root span named `adk.invocation`; tool calls and agent-to-agent transfers create child spans.

## Security best practices

| Practice | Implementation |
|---|---|
| API keys / secrets | `google.cloud.secretmanager`; inject as env vars at deploy time |
| Least-privilege IAM | Use per-agent service accounts; `roles/aiplatform.user` for Vertex; `roles/storage.objectViewer` for GCS artifacts |
| Input sanitisation | Use a `before_model_callback` plugin to strip PII before it reaches the model |
| Tool guardrails | `before_tool_callback` plugin that returns `{"error": "..."}` for blocked operations |
| Egress control | VPC Service Controls or a proxy allowlist for tool HTTP calls |
| Credential rotation | Store credentials in Secret Manager with version rotation; reference via `AuthConfig.credential_key` |

## Gotchas

- `BuiltInPlanner.thinking_config` takes precedence over `generate_content_config.thinking_config` on the same agent — a `UserWarning` is emitted if both are set.
- `PlanReActPlanner` requires the model to follow tag conventions (`/*PLANNING*/` etc.). Smaller or non-instruction-tuned models may not comply reliably.
- `BaseTool._get_declaration` returning `None` means no function schema is added to `LlmRequest.config.tools`. This is intentional for truly built-in (Gemini-side) tools — call `process_llm_request` instead to inject them.
- `BaseToolset.get_tools_with_prefix` is `@final` — override `get_tools` only.
- Authenticated tools that call `tool_context.request_credential` **pause** the current invocation. ADK stores the credential request in the event stream; the next user turn should carry the credential response.
- `AuthConfig.credential_key` defaults to a SHA-256 digest of the auth scheme + raw credential. Two toolsets with identical configs share the same stored credential automatically.

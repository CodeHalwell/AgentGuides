---
title: "Class deep dives — volume 8 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.1.0 classes: ReadonlyContext, FunctionNode, JoinNode/Trigger, ContainerCodeExecutor, GkeCodeExecutor, AgentEngineSandboxCodeExecutor, ApplicationIntegrationToolset, BigtableToolset/BigtableToolSettings, OpenAILlm."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 8"
  order: 67
---

Source-verified against **google-adk==2.1.0** (installed from PyPI, May 2026). Every field name, signature, and code example is taken directly from the installed package source.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `ReadonlyContext` | `google.adk.agents.readonly_context` | Stable |
| 2 | `FunctionNode` | `google.adk.workflow._function_node` | Stable |
| 3 | `JoinNode` + `Trigger` | `google.adk.workflow._join_node`, `._trigger` | Stable |
| 4 | `ContainerCodeExecutor` | `google.adk.code_executors.container_code_executor` | Stable |
| 5 | `GkeCodeExecutor` | `google.adk.code_executors.gke_code_executor` | Stable |
| 6 | `AgentEngineSandboxCodeExecutor` | `google.adk.code_executors.agent_engine_sandbox_code_executor` | Stable |
| 7 | `ApplicationIntegrationToolset` | `google.adk.tools.application_integration_tool` | Stable |
| 8 | `BigtableToolset` + `BigtableToolSettings` | `google.adk.tools.bigtable` | Experimental |
| 9 | `OpenAILlm` | `google.adk.labs.openai._openai_llm` | Labs |

---

## 1 · `ReadonlyContext`

`ReadonlyContext` is the **read-only view of `InvocationContext`** passed to the `instruction` callable on `LlmAgent` and to `before_agent_callback`. It exposes only safe, non-mutating surfaces: session state (as an immutable `MappingProxyType`), invocation metadata, and credential lookup.

### Source location

```
google.adk.agents.readonly_context.ReadonlyContext
```

> **Note:** `google.adk.agents.callback_context.CallbackContext` is an alias for `google.adk.agents.context.Context` (the full mutable context). `ReadonlyContext` is a strictly lighter, read-only wrapper used specifically where mutation should be forbidden.

### Constructor

```python
class ReadonlyContext:
    def __init__(self, invocation_context: InvocationContext) -> None:
        self._invocation_context = invocation_context
```

`ReadonlyContext` is constructed internally by the framework — you never create one yourself.

### Properties (source-verified)

| Property | Type | Description |
|----------|------|-------------|
| `user_content` | `Optional[types.Content]` | The user message that triggered this invocation. Labelled `READONLY` in source. |
| `invocation_id` | `str` | The current invocation ID (format: `"e-" + uuid`). |
| `agent_name` | `str` | Name of the currently running agent. |
| `state` | `MappingProxyType[str, Any]` | Current session state wrapped in `MappingProxyType` — **immutable**. Labelled `READONLY` in source. |
| `session` | `Session` | The active `Session` object (full access, but no state writes via proxy). |
| `user_id` | `str` | The ID of the current user. Labelled `READONLY` in source. |
| `run_config` | `Optional[RunConfig]` | Per-invocation config (max LLM calls, streaming, etc.). Labelled `READONLY` in source. |

### Methods

```python
def get_credential(self, key: str) -> Optional[AuthCredential]:
    """Gets a resolved credential by key for this invocation."""
    return self._invocation_context.credential_by_key.get(key)
```

`get_credential` lets your instruction callable inspect whether an auth credential was already resolved for a given key during this invocation.

### Where `ReadonlyContext` is used

```
LlmAgent(instruction=callable)
    └── callable receives ReadonlyContext
LlmAgent(before_model_callback=callable)
    └── callable receives ReadonlyContext (before model call)
BaseToolset.get_tools(readonly_context)
    └── receives ReadonlyContext for tool filtering
```

### Example 1 — dynamic system instruction based on session state

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.runners import InMemoryRunner


def build_instruction(ctx: ReadonlyContext) -> str:
    """Return a different system prompt based on the user's subscription tier."""
    tier = ctx.state.get("subscription_tier", "free")
    name = ctx.state.get("user_name", "user")

    base = f"You are a helpful assistant. The user's name is {name}."
    if tier == "pro":
        return base + " Provide detailed, technical answers."
    elif tier == "enterprise":
        return base + " Provide detailed answers with cost/compliance context."
    else:
        return base + " Keep answers concise and beginner-friendly."


agent = LlmAgent(
    name="adaptive_assistant",
    model="gemini-2.5-flash",
    instruction=build_instruction,  # callable receives ReadonlyContext
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="tiers")

    # Create a pro-tier session
    await runner.session_service.create_session(
        app_name="tiers",
        user_id="alice",
        session_id="pro_session",
        state={"subscription_tier": "pro", "user_name": "Alice"},
    )

    events = await runner.run_debug(
        "Explain database indexing.",
        user_id="alice",
        session_id="pro_session",
    )
    print(events[-1].content.parts[0].text[:300])


asyncio.run(main())
```

### Example 2 — tool filtering with `ReadonlyContext`

`BaseToolset.get_tools()` receives a `ReadonlyContext` so you can expose different tools per user without creating multiple agents:

```python
import asyncio
from typing import Optional, List
from google.adk.agents import LlmAgent
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.base_tool import BaseTool
from google.adk.tools import FunctionTool
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.runners import InMemoryRunner


def admin_delete(resource_id: str) -> str:
    """Delete a resource (admin only)."""
    return f"Deleted resource {resource_id}"


def read_resource(resource_id: str) -> str:
    """Read a resource."""
    return f"Resource {resource_id}: content here"


class RoleBasedToolset(BaseToolset):
    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> List[BaseTool]:
        tools = [FunctionTool(func=read_resource)]
        if readonly_context and readonly_context.state.get("role") == "admin":
            tools.append(FunctionTool(func=admin_delete))
        return tools

    async def close(self) -> None:
        pass


agent = LlmAgent(
    name="rbac_agent",
    model="gemini-2.5-flash",
    instruction="You are a resource management assistant.",
    tools=[RoleBasedToolset()],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="rbac")

    await runner.session_service.create_session(
        app_name="rbac", user_id="admin_user", session_id="s1",
        state={"role": "admin"},
    )
    await runner.session_service.create_session(
        app_name="rbac", user_id="regular_user", session_id="s2",
        state={"role": "viewer"},
    )

    print("Admin tools:", [
        t.name for t in await RoleBasedToolset().get_tools(None)
    ])


asyncio.run(main())
```

### Example 3 — reading user content in an instruction callable

```python
from google.adk.agents.readonly_context import ReadonlyContext


def smart_instruction(ctx: ReadonlyContext) -> str:
    """Adjust verbosity based on whether the user asked a simple question."""
    text = ""
    if ctx.user_content and ctx.user_content.parts:
        text = (ctx.user_content.parts[0].text or "").lower()

    word_count = len(text.split())
    if word_count <= 5:
        return "Answer in one sentence."
    elif "explain" in text or "how does" in text:
        return "Give a thorough explanation with examples."
    else:
        return "Answer concisely but completely."
```

---

## 2 · `FunctionNode`

`FunctionNode` wraps a **Python sync/async function or generator** as a first-class workflow node. It is the primary building block for custom logic in `Workflow` graphs and handles all the plumbing: parameter binding from session state or node input, type coercion with Pydantic, schema inference, and the HITL auth gate.

### Source location

```
google.adk.workflow._function_node.FunctionNode
```

`FunctionNode` inherits from `BaseNode` and is the class the `@node` decorator creates under the hood.

### Constructor (source-verified)

```python
class FunctionNode(BaseNode):
    def __init__(
        self,
        *,
        func: Callable[..., Any],
        name: str | None = None,
        rerun_on_resume: bool = False,
        retry_config: RetryConfig | None = None,
        timeout: float | None = None,
        auth_config: AuthConfig | None = None,
        parameter_binding: Literal['state', 'node_input'] = 'state',
        state_schema: type[BaseModel] | None = None,
    )
```

### Fields (source-verified)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auth_config` | `AuthConfig \| None` | `None` | Triggers a HITL auth request before the node runs. Requires `rerun_on_resume=True`. |
| `parameter_binding` | `Literal['state', 'node_input']` | `'state'` | `'state'`: parameters are looked up in `ctx.state`. `'node_input'`: parameters come from `node_input` dict; input/output schemas are inferred from the function signature. |

### Type coercions applied automatically

| Input type | Target annotation | Coercion |
|------------|-------------------|----------|
| `dict` | Pydantic `BaseModel` | `TypeAdapter(model).validate_python(dict)` |
| `list[dict]` | `list[BaseModel]` | element-wise |
| `types.Content` | `str` / `Optional[str]` | extracts `.parts[*].text` |
| Any | Any annotated type | `TypeAdapter(hint).validate_python(value)` |

### Parameter detection rules (from source)

1. A parameter named `ctx` (or any name whose type annotation is `Context`) is detected as the context parameter and bound to the live `Context` object.
2. In `'state'` mode, `node_input` is passed through directly (with coercion); all other parameters are looked up in `ctx.state`.
3. In `'node_input'` mode, all non-context parameters are looked up in the `node_input` dict; `input_schema` and `output_schema` are inferred from the function signature.

### Generator support

`FunctionNode` transparently handles all four callable flavours:

| Callable type | How it works |
|---------------|--------------|
| Sync function | Called directly; return value wrapped in `Event(output=...)` |
| Async function | `await`-ed; return value wrapped in `Event(output=...)` |
| Sync generator | Wrapped with `_sync_to_async_gen`; each yielded item becomes an `Event` |
| Async generator | Iterated natively; each yielded item becomes an `Event` |

### Example 1 — basic state-binding node

```python
import asyncio
from pydantic import BaseModel
from google.adk.workflow import Workflow
from google.adk.workflow._function_node import FunctionNode
from google.adk.agents.context import Context
from google.adk.runners import InMemoryRunner
from google.adk.agents import LlmAgent


class SummaryOutput(BaseModel):
    summary: str
    word_count: int


def summarise(ctx: Context, text: str) -> SummaryOutput:
    """Read 'text' from state, return a structured summary."""
    words = text.split()
    return SummaryOutput(
        summary=" ".join(words[:10]) + ("..." if len(words) > 10 else ""),
        word_count=len(words),
    )


summarise_node = FunctionNode(func=summarise, name="summarise")
```

### Example 2 — async generator node that streams events

```python
from google.adk.workflow._function_node import FunctionNode
from google.adk.agents.context import Context
from google.adk.events.event import Event
from typing import AsyncGenerator


async def chunk_processor(
    ctx: Context,
    items: list,
) -> AsyncGenerator[Event, None]:
    """Yield one event per item; accumulate results in state."""
    processed = []
    for item in items:
        result = item.strip().upper()
        processed.append(result)
        yield Event(
            output={"current": result},
            state={"last_processed": result},
        )
    # final summary event
    yield Event(output={"all": processed}, state={"processed_items": processed})


processor_node = FunctionNode(func=chunk_processor, name="chunk_processor")
```

### Example 3 — `node_input` binding with schema inference

Use `parameter_binding='node_input'` when a node acts as an agent tool or when you want explicit typed input rather than reading from state:

```python
from pydantic import BaseModel
from google.adk.workflow._function_node import FunctionNode
from google.adk.agents.context import Context


class ScoreInput(BaseModel):
    candidate: str
    score: float
    threshold: float


class ScoreOutput(BaseModel):
    passed: bool
    message: str


def evaluate(ctx: Context, node_input: ScoreInput) -> ScoreOutput:
    passed = node_input.score >= node_input.threshold
    msg = f"{node_input.candidate} {'passed' if passed else 'failed'} (score={node_input.score})"
    return ScoreOutput(passed=passed, message=msg)


eval_node = FunctionNode(
    func=evaluate,
    name="evaluate",
    parameter_binding="node_input",
)
# eval_node.input_schema  → inferred from ScoreInput
# eval_node.output_schema → inferred from ScoreOutput
```

### Example 4 — `auth_config` gate (HITL credential request)

```python
from google.adk.workflow._function_node import FunctionNode
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import OAuthGrantType, OpenIdConnectWithConfig
from google.adk.agents.context import Context


auth_cfg = AuthConfig(
    auth_scheme=OpenIdConnectWithConfig(
        authorization_endpoint="https://accounts.google.com/o/oauth2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/calendar.readonly"],
    ),
)


def fetch_calendar(ctx: Context) -> dict:
    """Fetch calendar events — only runs after credentials are provided."""
    cred = ctx.state.get("adk_auth_credential:calendar_auth")
    # use cred.oauth2.access_token to call Calendar API
    return {"events": ["Meeting 9am", "Standup 10am"]}


calendar_node = FunctionNode(
    func=fetch_calendar,
    name="fetch_calendar",
    rerun_on_resume=True,   # required when auth_config is set
    auth_config=auth_cfg,
)
```

---

## 3 · `JoinNode` + `Trigger`

### `JoinNode`

`JoinNode` is the synchronisation primitive for **fork/join patterns** in `Workflow` graphs. Unlike a regular node that fires when any predecessor sends a trigger, `JoinNode` waits until **all** upstream predecessors have sent their trigger before it runs.

```
google.adk.workflow._join_node.JoinNode
```

#### Source definition (key override)

```python
class JoinNode(BaseNode):
    @property
    @override
    def _requires_all_predecessors(self) -> bool:
        return True

    @override
    async def _run_impl(self, *, ctx: Context, node_input: Any):
        # Passes the aggregated dict of all predecessor outputs downstream
        yield Event(
            output=node_input,  # dict keyed by predecessor node names
            branch=ctx._invocation_context.branch,
        )
```

`JoinNode` simply passes through the **aggregated `node_input`** — a dict where each key is a predecessor node name and each value is that node's output. There is no custom logic; it is purely a synchronisation barrier.

#### Input validation with `input_schema`

When `input_schema` is set, `JoinNode` validates each predecessor's contribution individually:

```python
# From source: _validate_input_data override
if self.input_schema and isinstance(data, dict):
    return {k: self._validate_schema(v, self.input_schema) for k, v in data.items()}
```

### `Trigger`

`Trigger` is the **edge-level data envelope** that carries a node's output to its downstream nodes. It is an internal framework type — you do not construct `Trigger` objects directly; the workflow engine creates them.

```
google.adk.workflow._trigger.Trigger
```

#### Fields (source-verified)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | `Any` | `None` | The payload to pass to the downstream node. |
| `use_sub_branch` | `bool` | `False` | If `True`, the downstream node runs on a new sub-branch (used by parallel fanout). |
| `branch` | `str \| None` | `None` | The branch inherited from the predecessor. |
| `isolation_scope` | `str \| None` | `None` | Scope tag propagated to the triggered node. |

#### Note on serialisation

```python
model_config = ConfigDict(ser_json_bytes='base64')
```

`Trigger` serialises `bytes` payloads as base64 when checkpointed, so binary data (e.g. image bytes) flows through workflow edges without corruption.

### Complete fork/join example

```python
import asyncio
from google.adk.workflow import Workflow
from google.adk.workflow._function_node import FunctionNode
from google.adk.workflow._join_node import JoinNode
from google.adk.agents.context import Context
from google.adk.runners import InMemoryRunner
from google.adk.agents import LlmAgent


# Two parallel analysis branches
def sentiment_analysis(ctx: Context, text: str) -> dict:
    positive_words = {"good", "great", "excellent", "love", "happy"}
    words = set(text.lower().split())
    score = len(words & positive_words) / max(len(words), 1)
    return {"sentiment": "positive" if score > 0.1 else "negative", "score": score}


def keyword_extraction(ctx: Context, text: str) -> dict:
    stopwords = {"the", "a", "is", "in", "it", "of", "and"}
    words = [w for w in text.lower().split() if w not in stopwords]
    return {"keywords": words[:5]}


def merge_results(ctx: Context, node_input: dict) -> dict:
    """Combine results from both parallel branches."""
    sentiment = node_input.get("sentiment_branch", {})
    keywords = node_input.get("keyword_branch", {})
    return {
        "sentiment": sentiment.get("sentiment"),
        "score": sentiment.get("score"),
        "keywords": keywords.get("keywords", []),
    }


sentiment_node = FunctionNode(func=sentiment_analysis, name="sentiment_branch")
keyword_node = FunctionNode(func=keyword_extraction, name="keyword_branch")
join_node = JoinNode(name="join")
merge_node = FunctionNode(func=merge_results, name="merge_results")

wf = Workflow(
    name="text_analysis",
    nodes=[sentiment_node, keyword_node, join_node, merge_node],
    edges=[
        ("__start__", "sentiment_branch"),
        ("__start__", "keyword_branch"),
        ("sentiment_branch", "join"),
        ("keyword_branch", "join"),
        ("join", "merge_results"),
    ],
)

root_agent = LlmAgent(
    name="root",
    model="gemini-2.5-flash",
    instruction="You coordinate text analysis.",
    workflow=wf,
)
```

---

## 4 · `ContainerCodeExecutor`

`ContainerCodeExecutor` executes Python code inside a **Docker container**, providing isolation from the host without requiring cloud infrastructure. It uses the `docker` Python SDK to manage a long-lived container for the lifetime of the executor object.

### Source location

```
google.adk.code_executors.container_code_executor.ContainerCodeExecutor
```

### Key constraints (from source)

```python
# These are frozen fields — cannot be overridden:
stateful: bool = Field(default=False, frozen=True, exclude=True)
optimize_data_file: bool = Field(default=False, frozen=True, exclude=True)
```

`ContainerCodeExecutor` is always **stateless**: each code snippet runs in a fresh `exec_run` call in the same long-running container (the container's filesystem persists between calls, but no explicit state object is maintained). File-based optimisation is disabled because files cannot cross the exec boundary without explicit volume mounts.

### Constructor

```python
ContainerCodeExecutor(
    base_url: Optional[str] = None,    # Docker daemon URL (default: unix socket)
    image: Optional[str] = None,        # Pull/use this image tag
    docker_path: Optional[str] = None,  # Build from this Dockerfile directory
    **data,                             # BaseCodeExecutor fields
)
```

**One of `image` or `docker_path` must be set** — raises `ValueError` otherwise.

### Lifecycle

```
__init__
  ├── docker.from_env() (or DockerClient(base_url=...))
  ├── _build_docker_image()   ← only if docker_path is set
  ├── client.containers.run(image, detach=True, tty=True)
  └── _verify_python_installation()  ← exec_run(['which', 'python3'])

execute_code
  └── container.exec_run(['python3', '-c', code], demux=True)

atexit
  └── __cleanup_container()  ← container.stop() + container.remove()
```

### Example 1 — prebuilt image

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors.container_code_executor import ContainerCodeExecutor

# The image must have python3 installed; adk-code-executor:latest is the
# default ADK image if you build it from the ADK repo.
executor = ContainerCodeExecutor(image="python:3.11-slim")

agent = LlmAgent(
    name="code_agent",
    model="gemini-2.5-flash",
    instruction="You can execute Python code to answer questions.",
    code_executor=executor,
)
```

### Example 2 — build from Dockerfile

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors.container_code_executor import ContainerCodeExecutor

# Build from a local Dockerfile. The directory must contain a Dockerfile
# and have python3 installed in the resulting image.
executor = ContainerCodeExecutor(docker_path="./my_sandbox")

agent = LlmAgent(
    name="custom_sandbox_agent",
    model="gemini-2.5-flash",
    instruction="Run calculations in a sandboxed environment.",
    code_executor=executor,
)
```

A minimal `./my_sandbox/Dockerfile`:

```dockerfile
FROM python:3.11-slim
RUN pip install numpy pandas matplotlib
```

### Example 3 — remote Docker daemon

```python
from google.adk.code_executors.container_code_executor import ContainerCodeExecutor

# Connect to a remote Docker daemon (e.g., a hardened VM)
executor = ContainerCodeExecutor(
    base_url="tcp://sandbox-host:2376",
    image="my-registry/adk-sandbox:1.2",
)
```

### Execution flow (inside `execute_code`)

```python
exec_result = container.exec_run(
    ['python3', '-c', code_execution_input.code],
    demux=True,   # separates stdout and stderr into a tuple
)
stdout = exec_result.output[0].decode('utf-8') if exec_result.output[0] else ''
stderr = exec_result.output[1].decode('utf-8') if exec_result.output[1] else ''
return CodeExecutionResult(stdout=stdout, stderr=stderr, output_files=[])
```

`output_files` is always an empty list — the executor has no mechanism to extract generated files from the container. For file-producing code, use `GkeCodeExecutor` or `AgentEngineSandboxCodeExecutor` instead.

### Comparison with other executors

| Executor | Infrastructure | Stateful | Files | Security | Complexity |
|----------|---------------|----------|-------|----------|------------|
| `UnsafeLocalCodeExecutor` | Host process | Yes | Yes | None | Minimal |
| `ContainerCodeExecutor` | Local Docker | No | No | Container | Low |
| `GkeCodeExecutor` | GKE + gVisor | No | No | gVisor sandbox | Medium |
| `AgentEngineSandboxCodeExecutor` | Vertex AI Agent Engine | Yes (session) | Yes | Managed | Low-medium |

---

## 5 · `GkeCodeExecutor`

`GkeCodeExecutor` executes Python code in **gVisor-sandboxed Pods on GKE**, with two modes: `job` (creates a Kubernetes Job per execution) and `sandbox` (uses the Agent Sandbox client for persistent sandbox Pods).

### Source location

```
google.adk.code_executors.gke_code_executor.GkeCodeExecutor
```

### Fields (source-verified)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `namespace` | `str` | `"default"` | Kubernetes namespace for Jobs and ConfigMaps. |
| `image` | `str` | `"python:3.11-slim"` | Container image for the code runner Pod. |
| `timeout_seconds` | `int` | `300` | Watch timeout for Job completion. |
| `executor_type` | `Literal["job", "sandbox"]` | `"job"` | Execution mode. |
| `cpu_requested` | `str` | `"200m"` | CPU request for the Pod. |
| `mem_requested` | `str` | `"256Mi"` | Memory request. |
| `cpu_limit` | `str` | `"500m"` | CPU limit (max 0.5 cores). |
| `mem_limit` | `str` | `"512Mi"` | Memory limit. |
| `kubeconfig_path` | `str \| None` | `None` | Explicit kubeconfig file path. |
| `kubeconfig_context` | `str \| None` | `None` | Kubeconfig context name. |
| `sandbox_gateway_name` | `str \| None` | `None` | Sandbox gateway (sandbox mode only). |
| `sandbox_template` | `str \| None` | `"python-sandbox-template"` | Sandbox template (sandbox mode only). |

### Authentication priority (constructor)

```
1. Explicit kubeconfig_path / kubeconfig_context
2. In-cluster service account (when running inside GKE)
3. Default local ~/.kube/config
```

### RBAC requirements

The `ServiceAccount` used by the executor Pod needs these Kubernetes RBAC permissions:

```yaml
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["create", "delete", "get", "patch"]
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["get", "list", "watch", "create", "delete"]
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list"]
```

### Job execution flow (`executor_type="job"`)

```
execute_code (job mode)
  ├── _create_code_configmap(name, code)    → ConfigMap with code.py
  ├── _create_job_manifest(...)             → V1Job with gVisor runtime + resource limits
  ├── BatchV1Api.create_namespaced_job(...)
  ├── _add_owner_reference(job, configmap)  → auto-cleanup when job finishes
  └── _watch_job_completion(job_name)
        ├── Watch.stream(list_namespaced_job, field_selector=...)
        ├── on success → _get_pod_logs(job_name) → CodeExecutionResult(stdout=logs)
        └── on failure → _get_pod_logs(job_name) → CodeExecutionResult(stderr=logs)
```

Pod security context applied to every Job (from source):

```python
security_context=V1SecurityContext(
    run_as_non_root=True,
    run_as_user=1001,
    allow_privilege_escalation=False,
    read_only_root_filesystem=True,
    capabilities=V1Capabilities(drop=["ALL"]),
)
```

### Example 1 — job mode (default)

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors.gke_code_executor import GkeCodeExecutor

# Uses in-cluster service account or ~/.kube/config automatically.
executor = GkeCodeExecutor(
    namespace="adk-sandbox",
    image="python:3.11-slim",
    timeout_seconds=120,
    cpu_limit="1000m",
    mem_limit="1Gi",
)

agent = LlmAgent(
    name="secure_code_agent",
    model="gemini-2.5-flash",
    instruction="Execute Python in a gVisor-sandboxed GKE Pod.",
    code_executor=executor,
)
```

### Example 2 — job mode with explicit kubeconfig

```python
from google.adk.code_executors.gke_code_executor import GkeCodeExecutor

executor = GkeCodeExecutor(
    kubeconfig_path="/home/user/.kube/my-cluster-config",
    kubeconfig_context="gke_my-project_us-central1_my-cluster",
    namespace="adk-sandbox",
    image="gcr.io/my-project/adk-runner:latest",
)
```

### Example 3 — sandbox mode

Sandbox mode requires the `k8s-agent-sandbox` package (`pip install google-adk[extensions]`) and the Agent Sandbox controller deployed in the cluster:

```python
from google.adk.code_executors.gke_code_executor import GkeCodeExecutor

executor = GkeCodeExecutor(
    namespace="adk-sandbox",
    executor_type="sandbox",
    sandbox_gateway_name="sandbox-gateway",
    sandbox_template="python-sandbox-template",
)
```

### TTL-based cleanup

Jobs are garbage-collected automatically after 10 minutes (`ttl_seconds_after_finished=600`). The ConfigMap is owned by the Job via `ownerReferences`, so it is deleted when the Job is cleaned up — no manual cleanup is needed.

---

## 6 · `AgentEngineSandboxCodeExecutor`

`AgentEngineSandboxCodeExecutor` is the **fully managed code execution option** — it provisions a sandbox environment on Vertex AI Agent Engine and executes code there, with file upload/download support and session-scoped sandbox reuse.

### Source location

```
google.adk.code_executors.agent_engine_sandbox_code_executor.AgentEngineSandboxCodeExecutor
```

### Fields (source-verified)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sandbox_resource_name` | `str \| None` | `None` | Re-use an existing sandbox. Format: `projects/.../reasoningEngines/.../sandboxEnvironments/...` |
| `agent_engine_resource_name` | `str \| None` | `None` | Agent Engine to create sandboxes within. Format: `projects/.../reasoningEngines/...` |

### Three initialisation paths (from source)

```
Case 1: sandbox_resource_name is provided
  → Use that exact sandbox. No Agent Engine lookup needed.

Case 2: neither provided
  → Auto-create an Agent Engine lazily on first execute_code() call.
  → Reads GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION env vars.

Case 3: agent_engine_resource_name is provided (but not sandbox_resource_name)
  → Create sandboxes within that Agent Engine on demand.
```

### Session state key

```python
invocation_context.session.state['sandbox_name']
```

The executor stores the sandbox resource name in session state so it is reused across invocations within the same session, avoiding the latency of creating a new sandbox each time.

### Sandbox lifecycle check

Before each execution, if `sandbox_name` is found in session state, the executor verifies it is still `STATE_RUNNING`:

```python
sandbox = client.agent_engines.sandboxes.get(name=sandbox_name)
if sandbox is None or sandbox.state != 'STATE_RUNNING':
    create_new_sandbox = True
```

`google.api_core.exceptions.NotFound` (404) is caught and handled by creating a new sandbox.

### File support

Unlike `ContainerCodeExecutor`, this executor supports **input and output files**:

```python
# Input files are sent with the code execution request:
input_data = {
    'code': code_execution_input.code,
    'files': [
        {'name': f.name, 'contents': f.content, 'mimeType': f.mime_type}
        for f in code_execution_input.input_files
    ],
}

# Output files arrive in the response alongside stdout/stderr:
# code_execution_response.outputs → (json stdout/stderr) + (binary files)
```

### Example 1 — auto-provisioned Agent Engine

```python
import os
from google.adk.agents import LlmAgent
from google.adk.code_executors.agent_engine_sandbox_code_executor import (
    AgentEngineSandboxCodeExecutor,
)

os.environ["GOOGLE_CLOUD_PROJECT"] = "my-gcp-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

# No resource names provided — ADK creates the Agent Engine + sandbox lazily.
executor = AgentEngineSandboxCodeExecutor()

agent = LlmAgent(
    name="managed_code_agent",
    model="gemini-2.5-flash",
    instruction="Use Python to solve data analysis tasks.",
    code_executor=executor,
)
```

### Example 2 — re-use an existing Agent Engine

```python
from google.adk.code_executors.agent_engine_sandbox_code_executor import (
    AgentEngineSandboxCodeExecutor,
)

executor = AgentEngineSandboxCodeExecutor(
    agent_engine_resource_name=(
        "projects/my-project/locations/us-central1/reasoningEngines/12345"
    ),
)
```

### Example 3 — pin to a specific long-lived sandbox

```python
from google.adk.code_executors.agent_engine_sandbox_code_executor import (
    AgentEngineSandboxCodeExecutor,
)

# Use a pre-warmed sandbox with your dependencies already installed.
executor = AgentEngineSandboxCodeExecutor(
    sandbox_resource_name=(
        "projects/my-project/locations/us-central1"
        "/reasoningEngines/12345/sandboxEnvironments/67890"
    ),
)
```

---

## 7 · `ApplicationIntegrationToolset`

`ApplicationIntegrationToolset` generates ADK tools from **Google Cloud Application Integration** (workflow triggers) or **Integration Connectors** (entity CRUD + actions). It converts the integration's OpenAPI spec into `RestApiTool` or `IntegrationConnectorTool` instances that the LLM can call directly.

### Source location

```
google.adk.tools.application_integration_tool.application_integration_toolset.ApplicationIntegrationToolset
```

### Two modes of operation

| Mode | When to use | Required params |
|------|-------------|-----------------|
| **Integration mode** | Trigger a Cloud Application Integration workflow | `integration`, optionally `triggers` |
| **Connector mode** | CRUD entities or call actions on an Integration Connector | `connection` + (`entity_operations` and/or `actions`) |

### Constructor (source-verified)

```python
ApplicationIntegrationToolset(
    project: str,
    location: str,
    connection_template_override: Optional[str] = None,
    integration: Optional[str] = None,
    triggers: Optional[List[str]] = None,
    connection: Optional[str] = None,
    entity_operations: Optional[str] = None,  # dict[str, list[str]] as JSON string
    actions: Optional[list[str]] = None,
    tool_name_prefix: Optional[str] = "",
    tool_instructions: Optional[str] = "",
    service_account_json: Optional[str] = None,  # SA credentials JSON string
    auth_scheme: Optional[AuthScheme] = None,
    auth_credential: Optional[AuthCredential] = None,
    tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
)
```

### Auth resolution (from source)

```
service_account_json provided?
  YES → ServiceAccountCredential + HTTPBearer scheme (explicit SA)
  NO  → AuthCredentialTypes.SERVICE_ACCOUNT + use_default_credential=True
        (uses Application Default Credentials)
```

`auth_scheme` and `auth_credential` let you inject OAuth2 user credentials for connectors that require end-user authentication (e.g. Salesforce, ServiceNow). The `authOverrideEnabled` flag on the connection must be `true` for user auth to take effect — otherwise the toolset logs a warning and falls back to SA credentials.

### Example 1 — trigger a Cloud Application Integration workflow

```python
from google.adk.agents import LlmAgent
from google.adk.tools.application_integration_tool import ApplicationIntegrationToolset

toolset = ApplicationIntegrationToolset(
    project="my-gcp-project",
    location="us-central1",
    integration="order-processing-integration",
    triggers=["api_trigger/process_order_trigger"],
    service_account_json='{"type":"service_account","project_id":"my-project",...}',
)

agent = LlmAgent(
    name="order_agent",
    model="gemini-2.5-flash",
    instruction="You process customer orders using the integration workflow.",
    tools=[toolset],
)
```

### Example 2 — connector entity operations + actions

```python
import json
from google.adk.tools.application_integration_tool import ApplicationIntegrationToolset

toolset = ApplicationIntegrationToolset(
    project="my-gcp-project",
    location="us-central1",
    connection="salesforce-prod",
    # empty list means all CRUD operations for that entity
    entity_operations=json.dumps({
        "Account": ["LIST", "GET", "CREATE", "UPDATE"],
        "Contact": [],   # all operations
    }),
    actions=["create_opportunity", "get_pipeline_summary"],
    tool_name_prefix="sf",
    tool_instructions="Always confirm before creating or updating records.",
)
```

### Example 3 — user OAuth2 credentials (auth override)

When a connector has `authOverrideEnabled=true` in its configuration, pass user credentials:

```python
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.auth.auth_schemes import OAuthGrantType, OAuthScheme
from google.adk.tools.application_integration_tool import ApplicationIntegrationToolset

user_credential = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id="my-client-id",
        client_secret="my-secret",
        auth_uri="https://my-connector.auth/authorize",
        token_uri="https://my-connector.auth/token",
        scopes=["read:data", "write:data"],
    ),
)

toolset = ApplicationIntegrationToolset(
    project="my-project",
    location="us-central1",
    connection="my-oauth2-connector",
    entity_operations='{"Record": ["LIST", "GET"]}',
    auth_credential=user_credential,
)
```

### Example 4 — tool name filtering

```python
from google.adk.tools.application_integration_tool import ApplicationIntegrationToolset

toolset = ApplicationIntegrationToolset(
    project="my-project",
    location="us-central1",
    connection="crm-connector",
    entity_operations='{"Lead": []}',
    tool_filter=["crm_list_lead", "crm_get_lead"],  # expose only read tools
)
```

---

## 8 · `BigtableToolset` + `BigtableToolSettings`

`BigtableToolset` is an **experimental** toolset that gives agents access to Cloud Bigtable via GoogleSQL queries and metadata APIs. It exposes 7 built-in tools.

### Source location

```
google.adk.tools.bigtable.bigtable_toolset.BigtableToolset
google.adk.tools.bigtable.settings.BigtableToolSettings
```

> **Experimental:** Both classes are decorated with `@experimental(FeatureName.BIGTABLE_TOOLSET)`. The API may change between ADK releases.

### The 7 built-in tools

| Tool name | Function | Description |
|-----------|----------|-------------|
| `bigtable_list_instances` | `metadata_tool.list_instances` | List all Bigtable instances in a project |
| `bigtable_get_instance_info` | `metadata_tool.get_instance_info` | Get details for a specific instance |
| `bigtable_list_tables` | `metadata_tool.list_tables` | List tables within an instance |
| `bigtable_get_table_info` | `metadata_tool.get_table_info` | Get schema details for a table |
| `bigtable_list_clusters` | `metadata_tool.list_clusters` | List clusters in an instance |
| `bigtable_get_cluster_info` | `metadata_tool.get_cluster_info` | Get details for a cluster |
| `bigtable_execute_sql` | `query_tool.execute_sql` | Execute a GoogleSQL query against a Bigtable table |

### Constructor

```python
BigtableToolset(
    *,
    tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
    credentials_config: Optional[BigtableCredentialsConfig] = None,
    bigtable_tool_settings: Optional[BigtableToolSettings] = None,
)
```

### `BigtableToolSettings`

```python
class BigtableToolSettings(BaseModel):
    max_query_result_rows: int = 50
    # Maximum number of rows returned from execute_sql. Prevents runaway queries.
```

### `execute_sql` function signature

```python
async def execute_sql(
    project_id: str,
    instance_id: str,
    query: str,
    credentials: Credentials,
    settings: BigtableToolSettings,
    tool_context: ToolContext,
    parameters: Dict[str, Any] | None = None,
    parameter_types: Dict[str, Any] | None = None,
) -> dict
```

The LLM calls this tool with `project_id`, `instance_id`, and `query`; the framework injects `credentials`, `settings`, and `tool_context` automatically.

### Example 1 — basic agent with Bigtable access

```python
from google.adk.agents import LlmAgent
from google.adk.tools.bigtable.bigtable_toolset import BigtableToolset
from google.adk.tools.bigtable.settings import BigtableToolSettings

toolset = BigtableToolset(
    bigtable_tool_settings=BigtableToolSettings(max_query_result_rows=100),
)

agent = LlmAgent(
    name="bigtable_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a data analyst with access to Cloud Bigtable. "
        "Use bigtable_list_instances to find instances, bigtable_list_tables "
        "to inspect schemas, and bigtable_execute_sql to run queries."
    ),
    tools=[toolset],
)
```

### Example 2 — exposing only query tools

```python
from google.adk.tools.bigtable.bigtable_toolset import BigtableToolset

# Restrict to read-only operations
toolset = BigtableToolset(
    tool_filter=[
        "bigtable_list_instances",
        "bigtable_list_tables",
        "bigtable_execute_sql",
    ],
)
```

### Example 3 — service account credentials

```python
from google.adk.tools.bigtable.bigtable_credentials import BigtableCredentialsConfig
from google.adk.tools.bigtable.bigtable_toolset import BigtableToolset

creds_config = BigtableCredentialsConfig(
    service_account_json='{"type":"service_account","project_id":"my-project",...}',
)

toolset = BigtableToolset(credentials_config=creds_config)
```

### Typical agent query flow

```
User: "List all tables in the analytics Bigtable instance"
  → LLM calls bigtable_list_instances(project_id="my-project")
  → LLM calls bigtable_list_tables(project_id="my-project", instance_id="analytics")
  → LLM summarises table list

User: "How many events happened yesterday?"
  → LLM calls bigtable_execute_sql(
        project_id="my-project",
        instance_id="analytics",
        query="SELECT COUNT(*) FROM events WHERE timestamp >= '2026-05-30'"
    )
  → LLM interprets the result dict and answers
```

---

## 9 · `OpenAILlm` (labs)

`OpenAILlm` is a **labs-tier** LLM backend that lets you use **OpenAI GPT-4o, o1, and o3** models with the full ADK agent framework — tools, streaming, structured output, and session management all work identically to Gemini.

### Source location

```
google.adk.labs.openai._openai_llm.OpenAILlm
```

> **Labs:** This module lives under `google.adk.labs` and is not covered by the ADK stability guarantee. Requires `pip install openai`.

### Class definition (source-verified)

```python
class OpenAILlm(BaseLlm):
    model: str = "gpt-4o"
    max_tokens: int = 4096

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"gpt-.*", r"o1-.*", r"o3-.*"]
```

The `supported_models` regexes are matched against the `model` string passed to `LlmAgent`. Matching is done by `LLMRegistry` — if the string matches `gpt-.*`, `o1-.*`, or `o3-.*`, `OpenAILlm` is resolved automatically once registered.

### Supported `GenerateContentConfig` fields

`OpenAILlm.generate_content_async` maps these `LlmRequest.config` fields to OpenAI's API:

| ADK field | OpenAI param |
|-----------|-------------|
| `system_instruction` | `messages[0].role="system"` |
| `tools[0].function_declarations` | `tools` array |
| `response_schema` (Pydantic model / dict) | `response_format.json_schema` with `strict=True` |
| `response_mime_type="application/json"` | `response_format.json_object` |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `stop_sequences` | `stop` |
| `max_output_tokens` | `max_tokens` (overrides the class-level default) |

### Streaming accumulation

The streaming path accumulates tool call fragments before yielding the final tool-use event:

```python
# Partial text is streamed immediately:
yield LlmResponse(content=Content(...), partial=True)

# Tool calls are accumulated by index across chunks, then yielded as one event:
for index in sorted(tool_calls_accumulated.keys()):
    part = Part.from_function_call(name=acc["name"], args=json.loads(acc["arguments"]))
    parts.append(part)
yield LlmResponse(content=Content(role="model", parts=parts), partial=False)
```

### Example 1 — basic GPT-4o agent

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.labs.openai._openai_llm import OpenAILlm
from google.adk.models.registry import LLMRegistry

# Register OpenAILlm so that "gpt-.*" model strings resolve correctly.
LLMRegistry.register(OpenAILlm)

os.environ["OPENAI_API_KEY"] = "sk-..."

agent = LlmAgent(
    name="gpt_agent",
    model="gpt-4o",
    instruction="You are a helpful assistant.",
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="gpt_demo")
    await runner.session_service.create_session(
        app_name="gpt_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug("What is 17 * 23?", user_id="u1", session_id="s1")
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

### Example 2 — GPT-4o with tools

```python
import asyncio, os
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.runners import InMemoryRunner
from google.adk.labs.openai._openai_llm import OpenAILlm
from google.adk.models.registry import LLMRegistry

LLMRegistry.register(OpenAILlm)
os.environ["OPENAI_API_KEY"] = "sk-..."


def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    data = {"London": "12°C, cloudy", "New York": "22°C, sunny"}
    return data.get(city, f"No data for {city}")


agent = LlmAgent(
    name="weather_agent",
    model="gpt-4o",
    instruction="Answer weather questions using the get_weather tool.",
    tools=[FunctionTool(func=get_weather)],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="weather")
    await runner.session_service.create_session(
        app_name="weather", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What is the weather in London?", user_id="u1", session_id="s1"
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

### Example 3 — structured output with Pydantic

```python
import asyncio, os
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.labs.openai._openai_llm import OpenAILlm
from google.adk.models.registry import LLMRegistry

LLMRegistry.register(OpenAILlm)
os.environ["OPENAI_API_KEY"] = "sk-..."


class Sentiment(BaseModel):
    label: str       # "positive" | "negative" | "neutral"
    confidence: float


agent = LlmAgent(
    name="sentiment_agent",
    model="gpt-4o",
    instruction="Classify the sentiment of the user's message.",
    output_schema=Sentiment,
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="sentiment")
    await runner.session_service.create_session(
        app_name="sentiment", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "This product is absolutely fantastic!",
        user_id="u1",
        session_id="s1",
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

### Example 4 — multi-model team (Gemini orchestrator + GPT-4o worker)

```python
import asyncio, os
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from google.adk.runners import InMemoryRunner
from google.adk.labs.openai._openai_llm import OpenAILlm
from google.adk.models.registry import LLMRegistry

LLMRegistry.register(OpenAILlm)
os.environ["OPENAI_API_KEY"] = "sk-..."

# Creative writing specialist powered by GPT-4o
writer = LlmAgent(
    name="writer",
    model="gpt-4o",
    instruction="You are a creative writing specialist. Write vivid, engaging prose.",
)

# Gemini orchestrator delegates writing tasks to the GPT-4o worker
orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-flash",
    instruction=(
        "You are a content manager. For creative writing tasks, "
        "delegate to the writer agent."
    ),
    tools=[AgentTool(agent=writer)],
)


async def main():
    runner = InMemoryRunner(agent=orchestrator, app_name="multimodel")
    await runner.session_service.create_session(
        app_name="multimodel", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Write a two-sentence product description for a noise-cancelling headset.",
        user_id="u1",
        session_id="s1",
    )
    print(events[-1].content.parts[0].text)


asyncio.run(main())
```

### `o1` / `o3` reasoning models

The `supported_models` patterns include `o1-.*` and `o3-.*`. These models have different parameter support (no streaming, no `temperature`):

```python
import asyncio, os
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.labs.openai._openai_llm import OpenAILlm
from google.adk.models.registry import LLMRegistry

LLMRegistry.register(OpenAILlm)
os.environ["OPENAI_API_KEY"] = "sk-..."

# o3-mini for complex reasoning tasks
agent = LlmAgent(
    name="reasoner",
    model="o3-mini",
    instruction="Solve the given problem step by step.",
    # Note: o1/o3 models do not support streaming or temperature
)
```

---

## Summary: when to use each class

| Class | Primary use case | Key constraint |
|-------|-----------------|----------------|
| `ReadonlyContext` | Dynamic `instruction` callables; role-based tool filtering | Read-only — cannot mutate state |
| `FunctionNode` | Custom Python logic in `Workflow` graphs | Wraps any sync/async function or generator |
| `JoinNode` | Synchronising parallel workflow branches | Waits for ALL predecessors before firing |
| `Trigger` | Internal edge payload (framework-managed) | Not constructed directly |
| `ContainerCodeExecutor` | Local Docker sandbox; no cloud dependency | Stateless; no file output support |
| `GkeCodeExecutor` | Production GKE sandboxing with gVisor | Requires RBAC setup; `kubernetes` package |
| `AgentEngineSandboxCodeExecutor` | Fully managed cloud code execution with file I/O | Requires Vertex AI Agent Engine; auto-created if not provided |
| `ApplicationIntegrationToolset` | Integrate with GCP Application Integration or Connectors | SA credentials required; `authOverrideEnabled` for user auth |
| `BigtableToolset` | Query Bigtable via GoogleSQL from agents | Experimental; `max_query_result_rows` to cap results |
| `OpenAILlm` | Use GPT-4o / o1 / o3 models in ADK agents | Labs; `pip install openai`; register with `LLMRegistry` |

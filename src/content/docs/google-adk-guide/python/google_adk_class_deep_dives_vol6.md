---
title: "Class Deep Dives Vol. 6 — v2.11.0"
description: "Source-verified deep dives for 10 entries in google-adk 2.11.0: Workflow, Graph, BaseNode, FunctionNode, RetryConfig, App, LlmAgent advanced fields, OpenAPIToolset, RemoteA2aAgent, GcsArtifactService, PubSubToolset, and ToolboxToolset."
framework: google-adk
language: python
sidebar:
  order: 136
---

All examples and field tables on this page are source-verified against **google-adk==2.11.0**. The ten entries span the **workflow engine** (the DAG orchestrator that replaces LoopAgent and ParallelAgent), the **application container** (`App` with compaction and resumability), **advanced LlmAgent fields**, and toolsets/services covering REST APIs (`OpenAPIToolset`), A2A peers (`RemoteA2aAgent`), GCS artifacts (`GcsArtifactService`), Pub/Sub (`PubSubToolset`), and the MCP Toolbox (`ToolboxToolset`).

| # | Class / Symbol | Module | Subject |
|---|---|---|---|
| 1 | `Workflow` + `Graph` | `google.adk.workflow._workflow` / `._graph` | DAG orchestrator and compiled graph |
| 2 | `BaseNode` + `FunctionNode` | `google.adk.workflow._base_node` / `._function_node` | Node primitives and Python-function nodes |
| 3 | `RetryConfig` | `google.adk.workflow._retry_config` | Per-node retry policy |
| 4 | `App` + `EventsCompactionConfig` + `ResumabilityConfig` | `google.adk.apps.app` / `._configs` | Application container with compaction and resumability |
| 5 | `LlmAgent` — advanced fields | `google.adk.agents.llm_agent` | `static_instruction`, `mode`, `parallel_worker`, `output_schema`, `output_key` |
| 6 | `OpenAPIToolset` | `google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset` | Auto-generates tools from an OpenAPI 3.x spec |
| 7 | `RemoteA2aAgent` | `google.adk.a2a.agent._remote_a2a_agent` | Calls a remote agent over the A2A protocol |
| 8 | `GcsArtifactService` | `google.adk.artifacts.gcs_artifact_service` | GCS-backed artifact storage with versioning |
| 9 | `PubSubToolset` | `google.adk.tools.pubsub.pubsub_toolset` | Publish / pull / acknowledge Pub/Sub messages (experimental) |
| 10 | `ToolboxToolset` | `google.adk.tools.toolbox_toolset` | MCP Toolbox — connects agents to databases and external services |

---

## 1 — `Workflow` + `Graph`

**Modules:** `google.adk.workflow._workflow`, `google.adk.workflow._graph`

`Workflow` is a `BaseNode` that acts as the graph orchestration loop.  It is the **replacement for `SequentialAgent`, `ParallelAgent`, and `LoopAgent`** (all three are deprecated).  Unlike those agents, a `Workflow` is not a `BaseAgent` — it is wired to a `Runner` via `App(root_agent=workflow)`.

`Graph` is the compiled representation built from `edges`.  You never construct it directly — `Workflow.model_post_init` calls `Graph.from_edge_items(edges)` and runs `validate_graph()` automatically.

### `Workflow` field reference

Source-verified from `google/adk/workflow/_workflow.py`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `name` | `str` | required | Unique node identifier (must be a valid Python identifier) |
| `edges` | `list[EdgeItem]` | `[]` | Declarative edge list — the sole required input for most workflows |
| `graph` | `Graph \| None` | `None` | Compiled graph (populated by `model_post_init` from `edges`) |
| `max_concurrency` | `int \| None` | `None` | Cap on simultaneously running static nodes (`None` = unlimited) |
| `rerun_on_resume` | `bool` | `True` | Workflow reruns its orchestration loop on resume by default |
| `state_schema` | `type[BaseModel] \| None` | `None` | Validates `ctx.state` mutations against a Pydantic model |
| `retry_config` | `RetryConfig \| None` | `None` | Retries the whole sub-workflow on failure |
| `timeout` | `float \| None` | `None` | Seconds before the workflow is cancelled and treated as a failure |

### `Graph` internals

Source-verified from `google/adk/workflow/_graph.py`:

| Symbol | Purpose |
|---|---|
| `Edge` | Single directed edge: `from_node`, `to_node`, optional `route` |
| `RouteValue` | `bool \| int \| str` — the value a node emits to select a downstream edge |
| `RoutingMap` | `dict[RouteValue, NodeLike \| tuple[NodeLike, ...]]` — maps route values to one or more target nodes |
| `DEFAULT_ROUTE` | String sentinel `"__DEFAULT__"` — followed when no specific route matches |
| `NodeLike` | `BaseNode \| BaseTool \| Callable \| "START"` — anything `build_node()` can wrap |
| `EdgeItem` | `Edge \| tuple[ChainElement, ...]` — the union type stored in `Workflow.edges` |

`Graph.get_next_pending_nodes` implements the routing logic:
1. Edges with `route=None` are **always** followed (unconditional fan-out).
2. Specific route matches win over `DEFAULT_ROUTE`.
3. If routing edges exist but nothing matches, the branch ends with a warning.

### Minimal linear pipeline

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow, node, START

@node
def clean(node_input: str) -> str:
    return node_input.strip()

summarise = LlmAgent(
    name="summarise",
    model="gemini-2.5-flash",
    instruction="Summarise the input in one sentence.",
    mode="single_turn",
)

refine = LlmAgent(
    name="refine",
    model="gemini-2.5-flash",
    instruction="Make the summary punchier.",
    mode="single_turn",
)

pipeline = Workflow(
    name="summary_pipeline",
    edges=[(START, clean, summarise, refine)],
)

app = App(name="summariser", root_agent=pipeline)
runner = InMemoryRunner(app=app)

async def main():
    session = await runner.session_service.create_session(
        app_name="summariser", user_id="u1"
    )
    from google.genai import types as genai_types
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text="  Large language models are transformer-based neural networks.  ")],
        ),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Conditional routing with `RoutingMap`

A node sets `ctx.route` to a string value and the `RoutingMap` selects the next branch based on that value:

```python
from google.adk.workflow import Workflow, node, START

@node
def classify(ctx, node_input: str) -> str:
    """Sets ctx.route to 'positive', 'negative', or 'neutral'."""
    low = node_input.lower()
    if any(w in low for w in ("good", "great", "excellent")):
        ctx.route = "positive"
    elif any(w in low for w in ("bad", "terrible", "awful")):
        ctx.route = "negative"
    else:
        ctx.route = "neutral"
    return node_input  # pass original text to the selected handler

@node
def handle_positive(node_input: str) -> str:
    return f"[POSITIVE] {node_input}"

@node
def handle_negative(node_input: str) -> str:
    return f"[NEGATIVE] {node_input}"

@node
def handle_neutral(node_input: str) -> str:
    return f"[NEUTRAL] {node_input}"

# RoutingMap syntax: {route_value: target_node, ...}
routing_map = {
    "positive": handle_positive,
    "negative": handle_negative,
    "neutral": handle_neutral,
}

workflow = Workflow(
    name="sentiment_router",
    edges=[(START, classify, routing_map)],
)
```

### Fan-out / fan-in with `JoinNode`

```python
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, node, START
from google.adk.workflow import JoinNode  # waits for all predecessors

translate_fr = LlmAgent(
    name="translate_fr",
    model="gemini-2.5-flash",
    instruction="Translate the input to French.",
    mode="single_turn",
)
translate_de = LlmAgent(
    name="translate_de",
    model="gemini-2.5-flash",
    instruction="Translate the input to German.",
    mode="single_turn",
)

joiner = JoinNode(name="joiner")

@node
def combine(node_input: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in node_input.items())

workflow = Workflow(
    name="multi_translate",
    # START fans out to both translators; joiner waits for both
    edges=[
        (START, (translate_fr, translate_de)),   # tuple = fan-out
        (translate_fr, joiner),
        (translate_de, joiner),
        (joiner, combine),
    ],
)
```

### Concurrency cap

```python
# Allow at most 3 nodes to run simultaneously.
workflow = Workflow(
    name="bounded",
    edges=[...],
    max_concurrency=3,
)
```

---

## 2 — `BaseNode` + `FunctionNode`

**Modules:** `google.adk.workflow._base_node`, `google.adk.workflow._function_node`

`BaseNode` is the Pydantic abstract base for every object in the workflow graph.  `FunctionNode` is what `@node` / `build_node()` produces when wrapping a plain Python function or generator.

### `BaseNode` field reference

Source-verified from `google/adk/workflow/_base_node.py`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `name` | `str` | required | Must be a valid Python identifier |
| `description` | `str` | `""` | Human-readable; exposed to parent LlmAgent as tool description |
| `rerun_on_resume` | `bool` | `False` | Rerun from scratch on interrupt resume (default is to complete immediately) |
| `wait_for_output` | `bool` | `False` | Stay in WAITING state until a route/output is emitted (useful for `JoinNode`) |
| `retry_config` | `RetryConfig \| None` | `None` | Per-node retry policy |
| `timeout` | `float \| None` | `None` | Seconds before the node is cancelled |
| `input_schema` | `SchemaType \| None` | `None` | Validates/coerces `node_input` before `_run_impl` |
| `output_schema` | `SchemaType \| None` | `None` | Validates/coerces the output before downstream sees it |
| `state_schema` | `type[BaseModel] \| None` | `None` | Validates `ctx.state` mutations at runtime |

`BaseNode.run()` is decorated `@final` — subclasses implement `_run_impl()`.  The public `run()` wrapper applies input/output validation and normalises `None` → skip, `Event` → pass-through, `RequestInput` → interrupt, everything else → `Event(output=value)`.

### `FunctionNode` extras

Source-verified from `google/adk/workflow/_function_node.py`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `auth_config` | `AuthConfig \| None` | `None` | Triggers credential request before the first run; credential placed in `ctx.state` |
| `parameter_binding` | `"state" \| "node_input"` | `"state"` | How function params are bound: from `ctx.state` or from `node_input` dict |

`FunctionNode` automatically infers `input_schema` and `output_schema` from type hints when `parameter_binding="node_input"`.

**Type coercions applied by FunctionNode:**

| Input type | Annotation expects | Action |
|---|---|---|
| `dict` | Pydantic `BaseModel` | `TypeAdapter.validate_python()` coerces to model |
| `list[dict]` | `list[BaseModel]` | Element-wise coercion |
| `types.Content` | `str` | Text parts extracted and joined |
| anything | any | `TypeAdapter.validate_python()` |

### Binding from state (default)

```python
from pydantic import BaseModel
from google.adk.workflow import Workflow, node, START

class MyState(BaseModel):
    query: str = ""
    result: str = ""

@node  # parameter_binding="state" by default
def search(ctx, query: str) -> str:
    # 'query' is pulled from ctx.state["query"]
    hits = [f"Result for {query!r}"]
    ctx.state["result"] = hits[0]  # must be written to ctx.state explicitly
    return hits[0]

workflow = Workflow(
    name="search_flow",
    edges=[(START, search)],
    state_schema=MyState,
)
```

### Binding from node input

```python
from google.adk.workflow import node

@node(parameter_binding="node_input")
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

# The caller passes {"name": "Alice"} as node_input.
# FunctionNode infers input_schema={name: str, greeting: str}
# and output_schema=str from the signature.
```

### Async and generator functions

```python
import asyncio
from google.adk.workflow import node

@node
async def fetch_data(url: str) -> dict:
    await asyncio.sleep(0)  # non-blocking I/O
    return {"data": f"fetched from {url}"}

@node
async def stream_chunks(text: str):
    """Async generator: yields partial outputs as separate events."""
    for chunk in text.split():
        yield chunk
```

### Node with auth requirement

```python
from google.adk.auth.auth_config import AuthConfig
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.workflow import FunctionNode

oidc_scheme = OpenIdConnectWithConfig(
    authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
    token_endpoint="https://oauth2.googleapis.com/token",
    scopes=["openid", "email"],
)

def call_protected_api(ctx) -> str:
    from google.adk.auth.auth_handler import AuthHandler
    creds = AuthHandler(auth_config).get_auth_response(ctx.state)
    # Use creds.oauth2.access_token to call the API …
    return "api_result"

auth_config = AuthConfig(auth_scheme=oidc_scheme)
protected_node = FunctionNode(
    func=call_protected_api,
    name="protected_api",
    auth_config=auth_config,
    rerun_on_resume=True,  # required when auth_config is set
)
```

---

## 3 — `RetryConfig`

**Module:** `google.adk.workflow._retry_config`

`RetryConfig` attaches an exponential back-off retry policy to any `BaseNode`.  When set on a `Workflow`, the **entire sub-workflow** is retried (child nodes that already produced output or a state change are replayed from history; nodes that produced neither re-execute).

### Field reference

Source-verified from `google/adk/workflow/_retry_config.py`:

| Field | Type | Default when `None` | Purpose |
|---|---|---|---|
| `max_attempts` | `int \| None` | `5` | Total attempts including the first; `0` or `1` means no retries |
| `initial_delay` | `float \| None` | `1.0` s | Delay before the first retry |
| `max_delay` | `float \| None` | `60.0` s | Ceiling on the delay between retries |
| `backoff_factor` | `float \| None` | `2.0` | Delay multiplier applied after each failure |
| `jitter` | `float \| None` | `1.0` | Symmetric signed offset applied to the delay, drawn from `[-jitter * delay, +jitter * delay]`; `0.0` removes randomness |
| `exceptions` | `list[str \| type[BaseException]] \| None` | retry on all | Exception names or classes that trigger a retry |

The `exceptions` field accepts both class references and string names — the validator converts classes to names so the config remains JSON-serialisable.

### Examples

```python
from google.adk.workflow import RetryConfig, node, Workflow, START

# Retry up to 3 times on any exception (3 total attempts = 2 retries), with 2s → 4s back-off.
@node(retry_config=RetryConfig(max_attempts=3, initial_delay=2.0, backoff_factor=2.0))
def flaky_api_call(query: str) -> str:
    import random
    if random.random() < 0.5:
        raise ConnectionError("transient network error")
    return f"result for {query}"

workflow = Workflow(name="resilient", edges=[(START, flaky_api_call)])
```

```python
# Retry only on specific exceptions; give up fast.
from google.adk.workflow import RetryConfig

config = RetryConfig(
    max_attempts=4,
    initial_delay=0.5,
    max_delay=10.0,
    jitter=0.0,                     # deterministic delays
    exceptions=["TimeoutError", "ConnectionError"],
)
```

```python
# Retry a whole sub-workflow (on partial failure, already-done nodes replay from history).
from google.adk.workflow import Workflow, RetryConfig, START

sub_workflow = Workflow(
    name="etl",
    edges=[(START, extract_node, transform_node, load_node)],
    retry_config=RetryConfig(max_attempts=3, initial_delay=5.0),
)
```

---

## 4 — `App` + `EventsCompactionConfig` + `ResumabilityConfig`

**Module:** `google.adk.apps.app` + `google.adk.apps._configs`

`App` is the top-level Pydantic container that wraps a root node and application-wide plugins into a deployable unit.  The `InMemoryRunner`, `VertexAiRunner`, and the ADK dev-server all expect an `App` instance.

### `App` field reference

Source-verified from `google/adk/apps/app.py`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `name` | `str` | required | Must match `^[a-zA-Z][a-zA-Z0-9_-]*$`; `"user"` is reserved |
| `root_agent` | `BaseNode \| None` | required | Entry point — any `BaseNode`, including `LlmAgent` or `Workflow` |
| `plugins` | `list[BasePlugin]` | `[]` | App-wide plugins (e.g. `SaveFilesAsArtifactsPlugin`) |
| `events_compaction_config` | `EventsCompactionConfig \| None` | `None` | Token-based or sliding-window event compaction |
| `context_cache_config` | `ContextCacheConfig \| None` | `None` | Prompt-caching config applied to all LlmAgents |
| `resumability_config` | `ResumabilityConfig \| None` | `None` | Enables pause/resume across long-running invocations |

### `EventsCompactionConfig` field reference

Source-verified from `google/adk/apps/_configs.py`:

`EventsCompactionConfig` trims the session event history to keep context windows from growing unboundedly.  **At least one** complete trigger pair must be set (both pairs together are also valid):

| Trigger pair | Fields | Behaviour |
|---|---|---|
| Token-based | `token_threshold` (int, >0) + `event_retention_size` (int, ≥0) | Compacts after an invocation when the prompt token count ≥ threshold; keeps the last N raw events un-compacted |
| Sliding-window | `compaction_interval` (int, >0) + `overlap_size` (int, ≥0) | Compacts every N user-initiated invocations; overlaps by M invocations for continuity |

```python
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.agents import LlmAgent

agent = LlmAgent(name="agent", model="gemini-2.5-flash", instruction="Help the user.")

# Token-based compaction: summarise when prompt exceeds 50k tokens,
# keeping the last 20 raw events intact.
app = App(
    name="my_app",
    root_agent=agent,
    events_compaction_config=EventsCompactionConfig(
        token_threshold=50_000,
        event_retention_size=20,
    ),
)
```

```python
# Sliding-window: compact every 10 user turns, overlapping by 2.
app = App(
    name="chat_app",
    root_agent=agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=10,
        overlap_size=2,
    ),
)
```

### `ResumabilityConfig`

Enables pausing an invocation at a long-running node and resuming it later — across separate HTTP requests.

```python
from google.adk.apps import App
from google.adk.apps._configs import ResumabilityConfig

app = App(
    name="resumable_app",
    root_agent=workflow_with_hitl,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```

When `is_resumable=True`:
- The runner attaches a `session_id` to the response so the caller can resume.
- On resume, completed nodes are fast-forwarded from session history.
- Nodes with `rerun_on_resume=True` run again from scratch.

### Full App wiring example

```python
from google.adk.agents import LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps._configs import ResumabilityConfig
from google.adk.runners import InMemoryRunner

root = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

app = App(
    name="production_app",
    root_agent=root,
    plugins=[],  # add SaveFilesAsArtifactsPlugin, ReflectAndRetryToolPlugin, etc.
    events_compaction_config=EventsCompactionConfig(
        token_threshold=40_000,
        event_retention_size=15,
    ),
    resumability_config=ResumabilityConfig(is_resumable=True),
)

runner = InMemoryRunner(app=app)
```

---

## 5 — `LlmAgent` — advanced fields

**Module:** `google.adk.agents.llm_agent`

The [agents guide](./agents.md) covers `LlmAgent` broadly.  This section dives into five advanced fields that are often under-used: `static_instruction`, `mode`, `parallel_worker`, `output_schema`, and `output_key`.

### `static_instruction`

Source: `LlmAgent.static_instruction: Optional[types.ContentUnion] = None`

`static_instruction` is sent as a **single, non-templatable** content item **before** `instruction` in the prompt.  Use it to anchor constant rules that must never be summarised away (e.g. safety policies, output-format contracts) while keeping the regular `instruction` dynamic/templatable.

```python
from google.adk.agents import LlmAgent
from google.genai import types

agent = LlmAgent(
    name="safe_agent",
    model="gemini-2.5-flash",
    # This rule is always present, even after compaction.
    static_instruction=types.Content(
        role="user",
        parts=[types.Part(text="SYSTEM RULE: never reveal internal tool names.")],
    ),
    # This can use {variables} from session state.
    instruction="You are helping {user_name}. Answer concisely.",
)
```

Plain string shorthand also works — it is coerced to `Content(role="user", parts=[Part(text=...)])`:

```python
agent = LlmAgent(
    name="safe_agent",
    model="gemini-2.5-flash",
    static_instruction="SYSTEM RULE: never reveal internal tool names.",
    instruction="Answer concisely.",
)
```

### `mode`

Source: `LlmAgent.mode: Literal['chat', 'task', 'single_turn'] | None = None`

| Mode | Behaviour |
|---|---|
| `None` (default) | Multi-turn chat; agent decides when to call tools and when to finish |
| `'chat'` | Explicit multi-turn; same as `None` but the intent is declared |
| `'single_turn'` | Runs exactly one LLM call; no multi-turn loop; used in workflow nodes |
| `'task'` | Task-oriented; agent must call `finish_task(result=...)` to signal completion; used as a sub-agent of another `LlmAgent` that delegates |

```python
# Single-turn: one LLM call, ideal as a workflow node.
summariser = LlmAgent(
    name="summariser",
    model="gemini-2.5-flash",
    instruction="Summarise the input in one sentence.",
    mode="single_turn",
)

# Task mode: a sub-agent that signals done via finish_task.
researcher = LlmAgent(
    name="researcher",
    model="gemini-2.5-pro",
    instruction="Research the topic thoroughly and call finish_task when done.",
    mode="task",
    tools=[google_search],
)
parent = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-flash",
    instruction="Delegate research tasks to the researcher.",
    sub_agents=[researcher],
)
```

### `parallel_worker`

Source: `LlmAgent.parallel_worker: bool | None = None`

When `True`, the agent is wrapped in `_ParallelWorker` to process **list inputs** in parallel — each item in the list is processed by a separate worker. This is NOT for enabling concurrent static graph fan-out edges (those already run concurrently by default). Setting this to `True` on an agent that writes to session state that other parallel branches also write creates race conditions — use namespaced state keys (`"agent_name.key"`) to avoid collisions.

```python
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, START

# Static fan-out: no parallel_worker needed — Workflow runs branches concurrently.
reviewer_a = LlmAgent(
    name="reviewer_a",
    model="gemini-2.5-flash",
    instruction="Review the code for correctness.",
    mode="single_turn",
    output_key="review_a",  # writes to state["review_a"]
)
reviewer_b = LlmAgent(
    name="reviewer_b",
    model="gemini-2.5-flash",
    instruction="Review the code for performance.",
    mode="single_turn",
    output_key="review_b",  # writes to state["review_b"]
)

workflow = Workflow(
    name="dual_review",
    edges=[(START, (reviewer_a, reviewer_b))],  # fan-out — runs concurrently
)
```

### `output_schema` + `output_key`

`output_schema` constrains the LLM to return structured JSON, which ADK validates and exposes as the event's output.  `output_key` writes the validated output value to `ctx.state[output_key]` after each successful turn — when `output_schema` is set, this is the validated structured value (a dict); otherwise it is the final text response.

```python
from pydantic import BaseModel
from google.adk.agents import LlmAgent

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    reasoning: str

classifier = LlmAgent(
    name="classifier",
    model="gemini-2.5-flash",
    instruction="Classify the sentiment of the input text.",
    mode="single_turn",
    output_schema=SentimentResult,
    output_key="sentiment_result",  # also stored in ctx.state
)
```

Combining `output_schema` + `output_key` + workflow state-passing:

```python
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, node, START

class TopicList(BaseModel):
    topics: list[str]

extractor = LlmAgent(
    name="extractor",
    model="gemini-2.5-flash",
    instruction="Extract the main topics from the text as a JSON list.",
    mode="single_turn",
    output_schema=TopicList,
    output_key="topics_json",  # state["topics_json"] = {"topics": [...]}
)

@node
def format_topics(topics_json: dict) -> str:
    topics = topics_json.get("topics", [])
    return "\n".join(f"- {t}" for t in topics)

pipeline = Workflow(
    name="topic_pipeline",
    edges=[(START, extractor, format_topics)],
)
```

---

## 6 — `OpenAPIToolset`

**Module:** `google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset`

`OpenAPIToolset` parses an OpenAPI 3.x specification and creates one `RestApiTool` per operation.  The agent can call these tools like any other function tool; `RestApiTool` handles HTTP method, path, parameters, and request body automatically.

### Constructor reference

Source-verified from `openapi_toolset.py`:

| Parameter | Type | Purpose |
|---|---|---|
| `spec_dict` | `dict \| None` | Pre-parsed spec dictionary |
| `spec_str` | `str \| None` | Raw spec string (JSON or YAML) |
| `spec_str_type` | `"json" \| "yaml"` | Format of `spec_str` (default `"json"`) |
| `auth_scheme` | `AuthScheme \| None` | Applied to all generated tools |
| `auth_credential` | `AuthCredential \| None` | Applied to all generated tools |
| `credential_key` | `str \| None` | Stable key for credential caching across turns |
| `tool_filter` | `ToolPredicate \| list[str] \| None` | Include only selected tools |
| `tool_name_prefix` | `str \| None` | Prefix to prepend to every tool name (avoids collisions) |
| `ssl_verify` | `bool \| str \| ssl.SSLContext \| None` | Custom CA bundle or context for enterprise TLS proxies |
| `header_provider` | `Callable[[ReadonlyContext], dict[str, str]] \| None` | Dynamic per-request headers (e.g. correlation IDs, session tokens) |
| `httpx_client_factory` | `Callable[[], httpx.AsyncClient] \| None` | Custom HTTP client (proxies, HTTP/2, request signing) |
| `preserve_property_names` | `bool` | Keep camelCase names from spec instead of converting to snake_case |

### Basic usage

```python
import json
from google.adk.agents import LlmAgent
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

petstore_spec = json.loads(open("petstore.json").read())

toolset = OpenAPIToolset(spec_dict=petstore_spec)

# Pass the toolset directly; LlmAgent resolves tools at invocation time.
agent = LlmAgent(
    name="petstore_agent",
    model="gemini-2.5-flash",
    instruction="Help the user manage pets in the store.",
    tools=[toolset],
)
```

### From a YAML spec string

```python
import aiohttp
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

yaml_spec = """
openapi: "3.0.0"
info:
  title: Weather API
  version: "1.0"
paths:
  /weather:
    get:
      operationId: getWeather
      summary: Get current weather
      parameters:
        - name: city
          in: query
          required: true
          schema:
            type: string
      responses:
        "200":
          description: OK
"""

toolset = OpenAPIToolset(spec_str=yaml_spec, spec_str_type="yaml")
```

### With OAuth2 authentication

```python
from google.adk.auth.auth_credential import AuthCredential, OAuth2Auth, AuthCredentialTypes
from google.adk.auth.auth_schemes import OAuthGrantType, OpenIdConnectWithConfig
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

scheme = OpenIdConnectWithConfig(
    authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
    token_endpoint="https://oauth2.googleapis.com/token",
    scopes=["https://www.googleapis.com/auth/drive.readonly"],
)
credential = AuthCredential(
    auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,  # "openIdConnect"
    oauth2=OAuth2Auth(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
    ),
)

toolset = OpenAPIToolset(
    spec_dict=drive_spec,
    auth_scheme=scheme,
    auth_credential=credential,
    credential_key="drive_oauth",  # cached in session state under this key
)
```

### With dynamic headers and a custom HTTPS client

```python
import httpx
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

def get_headers(ctx: ReadonlyContext) -> dict[str, str]:
    """Inject a per-session correlation ID from state."""
    correlation_id = ctx.state.get("correlation_id", "default")
    return {"X-Correlation-Id": correlation_id}

def client_factory() -> httpx.AsyncClient:
    """Custom client with a corporate proxy (httpx 0.28+ uses proxy=)."""
    return httpx.AsyncClient(
        proxy="https://proxy.corp.example.com:8080",
        verify="/etc/ssl/certs/corp-ca.pem",
    )

toolset = OpenAPIToolset(
    spec_dict=internal_api_spec,
    header_provider=get_headers,
    httpx_client_factory=client_factory,
)
```

### Filtering tools from a large spec

```python
# Include only two specific operations by operationId (lowercased + underscored).
toolset = OpenAPIToolset(
    spec_dict=large_spec,
    tool_filter=["list_pets", "create_pet"],
    tool_name_prefix="petstore_",  # → "petstore_list_pets", "petstore_create_pet"
)

# Retrieve a single tool by name.
create_tool = toolset.get_tool("petstore_create_pet")
```

---

## 7 — `RemoteA2aAgent`

**Module:** `google.adk.a2a.agent._remote_a2a_agent` (re-exported from `google.adk.agents.remote_a2a_agent`)

`RemoteA2aAgent` is a `BaseAgent` subclass that calls a remote service over the [A2A (Agent-to-Agent) protocol](./mcp-and-a2a.md).  The remote service may be another ADK agent, a LangGraph agent, or any A2A-compliant server.

### Constructor reference

Source-verified from `_remote_a2a_agent.py`:

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `name` | `str` | required | Unique agent identifier |
| `agent_card` | `AgentCard \| str` | required | Agent card object, URL to card JSON, or file path |
| `description` | `str` | `""` | Auto-populated from the card if empty |
| `timeout` | `float` | `600.0` s | HTTP request timeout |
| `auth_scheme` | `AuthScheme \| None` | `None` | Auth applied to every request including card fetch |
| `auth_credential` | `AuthCredential \| None` | `None` | Credential for `auth_scheme` |
| `credential_key` | `str \| None` | `None` | Cache key for the resolved credential |
| `a2a_client_factory` | `A2AClientFactory \| None` | `None` | Custom A2A client (replaces deprecated `httpx_client`) |
| `full_history_when_stateless` | `bool` | `False` | Send full session history to stateless peers |
| `context_builder` | `Callable \| None` | `None` | Custom message construction logic |
| `use_legacy` | `bool` | `True` | `False` adds the new integration extension header |
| `mode` | `"task" \| None` | `None` | `"task"` for sub-agent delegation (peer must call `finish_task`) |

### Basic usage — URL-based card

```python
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

remote_agent = RemoteA2aAgent(
    name="weather_service",
    agent_card="https://weather.example.com/.well-known/agent.json",
    description="Provides current weather for any city.",
)

orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-flash",
    instruction="Use available agents to answer user queries.",
    sub_agents=[remote_agent],
)
```

### Loading a card from a local file

```python
remote_agent = RemoteA2aAgent(
    name="local_peer",
    agent_card="./agent_card.json",   # file path as string
)
```

### Building from an `AgentCard` object

```python
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

card = AgentCard(
    name="summariser",
    description="Summarises documents.",
    url="https://summariser.example.com/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    skills=[AgentSkill(id="summarise", name="Summarise", description="Summarise a document")],
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
)

remote_agent = RemoteA2aAgent(name="summariser", agent_card=card)
```

### Task-mode delegation

```python
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

class ResearchOutput(BaseModel):
    summary: str
    sources: list[str]

# In task mode, output_schema must be on the RemoteA2aAgent so ADK can
# unwrap the finish_task arguments into the typed result.
remote_researcher = RemoteA2aAgent(
    name="researcher",
    agent_card="https://research.example.com/.well-known/agent.json",
    mode="task",
    output_schema=ResearchOutput,
)

parent = LlmAgent(
    name="parent",
    model="gemini-2.5-flash",
    instruction="Delegate research to the researcher sub-agent.",
    sub_agents=[remote_researcher],
)
```

### With API key authentication

```python
from fastapi.openapi.models import HTTPBearer
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, HttpAuth, HttpCredentials,
)
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

scheme = HTTPBearer(scheme="bearer")  # fastapi.openapi.models.HTTPBearer; scheme field required
credential = AuthCredential(
    auth_type=AuthCredentialTypes.HTTP,  # "http"
    http=HttpAuth(
        scheme="bearer",  # required: the HTTP auth scheme name
        credentials=HttpCredentials(token="MY_API_KEY"),
    ),
)

remote_agent = RemoteA2aAgent(
    name="secure_peer",
    agent_card="https://secure-agent.example.com/.well-known/agent.json",
    auth_scheme=scheme,
    auth_credential=credential,
    credential_key="secure_peer_key",
)
```

---

## 8 — `GcsArtifactService`

**Module:** `google.adk.artifacts.gcs_artifact_service`

`GcsArtifactService` stores agent artifacts as versioned GCS blobs.  Each call to `save_artifact` appends a new version — the service never overwrites.

### Blob naming scheme (source-verified)

```
# Session-scoped file
{app_name}/{user_id}/{session_id}/{filename}/{version}

# User-scoped file (filename starts with "user:")
# NOTE: the "user:" prefix is preserved in the stored blob path — it is NOT stripped.
{app_name}/{user_id}/user/{filename}/{version}
# e.g. save_artifact("user:profile.json", ...) → my_app/u1/user/user:profile.json/0
```

Versions are integers starting from `0`.  Up to `_MAX_SAVE_VERSION_ATTEMPTS = 10` concurrent save attempts are retried to resolve version conflicts.

### Constructor reference

```python
class GcsArtifactService(BaseArtifactService):
    def __init__(self, bucket_name: str, **kwargs: Any): ...
```

`**kwargs` are forwarded to `google.cloud.storage.Client(...)`, so you can pass `project`, `credentials`, `client_options`, etc.

### Setup and basic usage

```python
from google.adk.artifacts.gcs_artifact_service import GcsArtifactService
from google.adk.runners import InMemoryRunner
from google.adk.apps import App
from google.adk.agents import LlmAgent

artifact_service = GcsArtifactService(bucket_name="my-project-artifacts")

app = App(name="my_app", root_agent=LlmAgent(name="agent", model="gemini-2.5-flash", instruction="Help."))
runner = InMemoryRunner(app=app, artifact_service=artifact_service)
```

### Saving an artifact from a tool

```python
from google.adk.tools.tool_context import ToolContext
from google.genai import types

async def save_report(report_text: str, tool_context: ToolContext) -> str:
    """Save a text report as an artifact."""
    artifact = types.Part.from_text(text=report_text)
    version = await tool_context.save_artifact(
        filename="report.txt",
        artifact=artifact,
    )
    return f"Saved report as version {version}"
```

### Loading a specific version

```python
async def load_report(version: int, tool_context: ToolContext) -> str:
    artifact = await tool_context.load_artifact(
        filename="report.txt",
        version=version,
    )
    return artifact.text if artifact else "Not found"
```

### User-scoped vs session-scoped

```python
# Session-scoped: visible only within this session.
await tool_context.save_artifact(filename="session_data.json", artifact=part)

# User-scoped: visible across all sessions for this user.
# Use the "user:" prefix in the filename.
await tool_context.save_artifact(filename="user:profile.json", artifact=part)
```

### Listing artifact keys

```python
async def list_my_artifacts(tool_context: ToolContext) -> list[str]:
    return await tool_context.list_artifacts()  # ToolContext method (not list_artifact_keys)
```

### Custom GCS credentials

```python
from google.oauth2 import service_account
from google.adk.artifacts.gcs_artifact_service import GcsArtifactService

credentials = service_account.Credentials.from_service_account_file(
    "service-account.json",
    scopes=["https://www.googleapis.com/auth/devstorage.read_write"],
)

artifact_service = GcsArtifactService(
    bucket_name="my-artifacts",
    project="my-gcp-project",
    credentials=credentials,
)
```

---

## 9 — `PubSubToolset` (experimental)

**Module:** `google.adk.tools.pubsub.pubsub_toolset`

`PubSubToolset` is an **experimental** toolset that exposes three Pub/Sub operations as agent tools: `publish_message`, `pull_messages`, and `acknowledge_messages`.  It requires `google-cloud-pubsub` to be installed.

> **Feature flag:** `PubSubToolset` is gated behind `FeatureName.PUBSUB_TOOLSET`.  Experimental features may change without notice.

### Constructor reference

Source-verified from `pubsub_toolset.py`:

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `tool_filter` | `ToolPredicate \| list[str] \| None` | `None` (all tools) | Include only selected tools |
| `credentials_config` | `PubSubCredentialsConfig \| None` | `None` | GCP credentials; uses Application Default Credentials if `None` |
| `pubsub_tool_config` | `PubSubToolConfig \| None` | `PubSubToolConfig()` | Project ID, timeouts, max messages, etc. |

Note: `tool_filter=None` (the default) selects **all** tools. `tool_filter=[]` selects **no** tools — an empty list is not the same as `None`.

### The three tools

| Tool function | Key arguments | Returns |
|---|---|---|
| `publish_message` | `topic_name`, `message`, `attributes?`, `ordering_key?` | `{"message_id": "..."}` or `{"status": "ERROR", ...}` |
| `pull_messages` | `subscription_name`, `max_messages?` | `{"messages": [{"message_id": ..., "data": ..., "ack_id": ...}]}` |
| `acknowledge_messages` | `subscription_name`, `ack_ids` | `{"status": "OK"}` or `{"status": "ERROR", ...}` |

### Basic setup

```python
from google.adk.agents import LlmAgent
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig

toolset = PubSubToolset(
    pubsub_tool_config=PubSubToolConfig(project_id="my-gcp-project"),
)

agent = LlmAgent(
    name="pubsub_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You can publish and pull Pub/Sub messages. "
        "Always acknowledge messages after processing them."
    ),
    tools=[toolset],  # pass toolset directly; tools resolved at invocation time
)
```

### Exposing only the publisher

```python
toolset = PubSubToolset(
    tool_filter=["publish_message"],
    pubsub_tool_config=PubSubToolConfig(project_id="my-project"),
)
```

### With explicit credentials

```python
from google.oauth2 import service_account
from google.adk.tools.pubsub.pubsub_credentials import PubSubCredentialsConfig

credentials = service_account.Credentials.from_service_account_file(
    "sa-key.json",
    scopes=["https://www.googleapis.com/auth/pubsub"],
)

toolset = PubSubToolset(
    credentials_config=PubSubCredentialsConfig(credentials=credentials),
    pubsub_tool_config=PubSubToolConfig(project_id="my-project"),
)
```

### Full publish-process-acknowledge pattern

```python
from google.adk.agents import LlmAgent
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
import asyncio

config = PubSubToolConfig(project_id="my-project")
toolset = PubSubToolset(pubsub_tool_config=config)

processor = LlmAgent(
    name="processor",
    model="gemini-2.5-flash",
    instruction="""
    You are a Pub/Sub message processor.
    1. Pull up to 5 messages from the subscription.
    2. For each message, extract the JSON payload and summarise it.
    3. Acknowledge all processed messages using their ack_ids.
    """,
    tools=[toolset],
)

app = App(name="pubsub_processor", root_agent=processor)
runner = InMemoryRunner(app=app)

async def process_subscription(subscription: str):
    session = await runner.session_service.create_session(
        app_name="pubsub_processor", user_id="worker"
    )
    async for event in runner.run_async(
        user_id="worker",
        session_id=session.id,
        new_message=f"Process messages from {subscription}",
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)
```

---

## 10 — `ToolboxToolset`

**Module:** `google.adk.tools.toolbox_toolset`

`ToolboxToolset` connects agents to a running [MCP Toolbox for Databases](https://github.com/googleapis/mcp-toolbox-sdk-python) server, which provides safe, structured access to databases (BigQuery, Spanner, AlloyDB, PostgreSQL, MySQL, etc.) and external services.

Requires `pip install google-adk[toolbox]` (installs `toolbox-adk`).

### Constructor reference

Source-verified from `toolbox_toolset.py`:

| Parameter | Type | Purpose |
|---|---|---|
| `server_url` | `str` | URL of the running MCP Toolbox server (e.g. `"http://127.0.0.1:5000"`) |
| `toolset_name` | `str \| None` | Named toolset to load from the server; `None` loads tools by name |
| `tool_names` | `list[str] \| None` | Specific tool names to expose; `None` loads all from `toolset_name` |
| `auth_token_getters` | `Mapping[str, TokenGetter] \| None` | Per-auth-service callables returning a bearer token |
| `bound_params` | `Mapping[str, object] \| None` | Parameter values bound at toolset creation (static or callable) |
| `credentials` | `CredentialConfig \| None` | `toolbox_adk.CredentialConfig` for service-to-service auth |
| `additional_headers` | `Mapping[str, HeaderValue] \| None` | Static headers forwarded to every tool request |
| `**kwargs` | | Additional arguments forwarded to `toolbox_adk.ToolboxToolset` |

If both `toolset_name` and `tool_names` are `None`, **all tools** registered on the server are loaded.  If both are set, tools from both sources are combined.

### Basic setup

```python
from google.adk.agents import LlmAgent
from google.adk.tools.toolbox_toolset import ToolboxToolset
from google.adk.apps import App
from google.adk.runners import InMemoryRunner

toolset = ToolboxToolset(server_url="http://127.0.0.1:5000")

# Pass the toolset directly; LlmAgent resolves tools at invocation time.
agent = LlmAgent(
    name="db_agent",
    model="gemini-2.5-flash",
    instruction="Help the user query and modify the database.",
    tools=[toolset],
)

app = App(name="db_app", root_agent=agent)
runner = InMemoryRunner(app=app)
```

### Loading a named toolset

```python
# The server exposes a "customer_tools" toolset in its config.
toolset = ToolboxToolset(
    server_url="http://127.0.0.1:5000",
    toolset_name="customer_tools",
)
```

### Loading specific tools

```python
toolset = ToolboxToolset(
    server_url="http://127.0.0.1:5000",
    tool_names=["search_customers", "update_customer_email"],
)
```

### Bound parameters (static context injection)

`bound_params` inject values into every tool call without the agent having to supply them — useful for tenant ID, user context, or environment flags:

```python
toolset = ToolboxToolset(
    server_url="http://127.0.0.1:5000",
    toolset_name="inventory",
    bound_params={
        "tenant_id": "acme_corp",
        "environment": "production",
    },
)
```

### Dynamic bound parameters from session state

Callables in `bound_params` must be **zero-argument** — they are called with no arguments each time a tool is invoked:

```python
def get_current_user_id() -> str:
    """Returns the current user ID from your application context."""
    return current_user.id  # e.g. from a thread-local or context variable

toolset = ToolboxToolset(
    server_url="http://127.0.0.1:5000",
    toolset_name="user_data",
    bound_params={
        "user_id": get_current_user_id,  # called with no args at invocation time
    },
)
```

### With authentication token getter

```python
import google.auth.transport.requests
import google.oauth2.id_token

def get_id_token() -> str:
    """Fetches a Google-signed ID token for service-to-service auth."""
    request = google.auth.transport.requests.Request()
    # fetch_id_token uses Application Default Credentials and the metadata server.
    return google.oauth2.id_token.fetch_id_token(request, audience="https://toolbox.internal.example.com")

toolset = ToolboxToolset(
    server_url="https://toolbox.internal.example.com",
    toolset_name="analytics",
    auth_token_getters={"google": get_id_token},
)
```

### Cleanup

Always call `close()` when done to release HTTP connections:

```python
async def main():
    toolset = ToolboxToolset(server_url="http://127.0.0.1:5000")
    try:
        agent = LlmAgent(
            name="agent",
            model="gemini-2.5-flash",
            instruction="Query the database.",
            tools=[toolset],  # toolset is a BaseToolset; resolved at invocation time
        )
        # ... run the agent ...
    finally:
        await toolset.close()
```

---

## Summary table

| # | Class | Key takeaway |
|---|---|---|
| 1 | `Workflow` + `Graph` | DAG orchestrator; `edges=[(START, a, b, c)]` builds the graph; `RoutingMap` enables conditional routing |
| 2 | `BaseNode` + `FunctionNode` | `parameter_binding="state"` reads from `ctx.state`; `"node_input"` reads from `node_input` dict; type coercions are automatic |
| 3 | `RetryConfig` | Attach to any node; defaults are `max_attempts=5`, `initial_delay=1.0 s`, `backoff_factor=2.0` when fields are `None` |
| 4 | `App` + configs | `EventsCompactionConfig` needs either `(token_threshold, event_retention_size)` or `(compaction_interval, overlap_size)` |
| 5 | `LlmAgent` advanced | `static_instruction` survives compaction; `mode="single_turn"` is best for workflow nodes; `output_key` writes to state |
| 6 | `OpenAPIToolset` | `header_provider` injects dynamic headers; `httpx_client_factory` replaces the default client entirely |
| 7 | `RemoteA2aAgent` | `agent_card` accepts a URL, file path, or `AgentCard` object; `mode="task"` requires the peer to call `finish_task` |
| 8 | `GcsArtifactService` | Blob naming: `app/user/session/filename/version`; `user:` prefix in filename creates user-scoped artifacts |
| 9 | `PubSubToolset` | Experimental; `tool_filter=None` exposes all 3 tools (publish/pull/ack); always acknowledge after processing |
| 10 | `ToolboxToolset` | `bound_params` accepts callables for dynamic values; `toolset_name + tool_names` can be combined |

## Version note

All field names, defaults, and call signatures are verified against **google-adk==2.11.0** installed from PyPI.  Check `pip show google-adk` to confirm your installed version.

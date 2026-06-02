---
title: "Class deep dives — volume 10 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.1.0 classes: Graph/Edge (workflow graph internals), NodeRunner (per-node executor), NodeState/NodeStatus (execution lifecycle), _ParallelWorker (fan-out parallel execution), ActiveStreamingTool/TranscriptionEntry (live audio agents), CachePerformanceAnalyzer (cache analytics), StreamingResponseAggregator (SSE/BIDI streaming engine), AgentRefConfig/ArgumentConfig/CodeConfig (YAML agent config DSL), BaseAuthProvider/AuthProviderRegistry (pluggable auth), FinishTaskTool/TaskRequest/TaskResult (task delegation protocol)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 10"
  order: 69
---

Source-verified against **google-adk==2.1.0** (installed from PyPI, June 2026). Every field name, signature, and code example is taken directly from the installed package source.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `Graph` + `Edge` | `google.adk.workflow._graph` | Stable |
| 2 | `NodeRunner` | `google.adk.workflow._node_runner` | Stable |
| 3 | `NodeState` + `NodeStatus` | `google.adk.workflow._node_state`, `_node_status` | Stable |
| 4 | `_ParallelWorker` | `google.adk.workflow._parallel_worker` | Stable |
| 5 | `ActiveStreamingTool` + `TranscriptionEntry` | `google.adk.agents.active_streaming_tool`, `transcription_entry` | Stable |
| 6 | `CachePerformanceAnalyzer` | `google.adk.utils.cache_performance_analyzer` | Experimental |
| 7 | `StreamingResponseAggregator` | `google.adk.utils.streaming_utils` | Stable |
| 8 | `AgentRefConfig` + `ArgumentConfig` + `CodeConfig` | `google.adk.agents.common_configs` | Experimental |
| 9 | `BaseAuthProvider` + `AuthProviderRegistry` | `google.adk.auth.base_auth_provider`, `auth_provider_registry` | Experimental |
| 10 | `FinishTaskTool` + `TaskRequest` + `TaskResult` + `_DefaultTaskOutput` | `google.adk.agents.llm.task` | Stable |

---

## 1 · `Graph` + `Edge`

**Source:** `google.adk.workflow._graph`

`Graph` is the compiled representation of a `Workflow`'s edge declarations. When you write `Workflow(edges=[...])`, `model_post_init` calls `_build_graph()`, which calls `Graph.from_edge_items()` to produce a validated `Graph`. Understanding `Graph` internals lets you reason about routing, reachability, and static schema checks before a single node runs.

### `Edge` — the unit of connectivity

```python
class Edge(BaseModel):
    from_node: Annotated[BaseNode, SerializeAsAny()]
    to_node:   Annotated[BaseNode, SerializeAsAny()]
    route: RouteValue | list[RouteValue] | None = None
```

`RouteValue` is any hashable value a node emits from `ctx.emit_route()`. Three forms:

| `route` value | Meaning |
|---|---|
| `None` | Unconditional — always followed when `from_node` completes |
| `"some_string"` or any scalar | Followed only when the node emits that exact value |
| `["a", "b"]` | Followed when the node emits any value in the list |
| `DEFAULT_ROUTE` | Fallback — followed when no specific route matched |

> `DEFAULT_ROUTE` **cannot** be combined with other routes in a list. One edge per `from_node` may carry `DEFAULT_ROUTE`; the validator raises if there are multiple.

### `Graph.from_edge_items()` — building the graph

```python
@classmethod
def from_edge_items(cls, edge_items: list[EdgeItem]) -> Graph:
    node_map: dict[int, BaseNode] = {}
    graph_edges: list[Edge] = []
    for item in edge_items:
        if isinstance(item, Edge):
            _process_explicit_edge(item, node_map, graph_edges)
        elif isinstance(item, tuple):
            _process_chain(item, node_map, graph_edges)
        else:
            raise ValueError(f"Invalid edge type: {type(item)}")
    return Graph(edges=graph_edges)
```

Two `EdgeItem` forms:
- **`Edge` object** — explicit `from_node → to_node` with optional `route`
- **`tuple`** — a chain `(A, B, C)` expands to `A → B → B → C` unconditionally; can mix nodes and routing maps

`_process_chain` also handles **routing maps** inside tuples:

```python
# Routing map: node A emits "yes" → B, "no" → C, DEFAULT_ROUTE → D
Workflow(edges=[
    (START, node_a, {"yes": node_b, "no": node_c, DEFAULT_ROUTE: node_d}),
])
```

### `Graph.validate_graph()` — the validation pipeline

Called automatically by `Workflow._build_graph()`. Six sequential checks:

| Check | What it catches |
|---|---|
| `_validate_duplicate_node_names()` | Two distinct node objects with the same name |
| `_validate_start_node()` | No `START` sentinel in the graph |
| `_validate_connectivity()` | Nodes unreachable from `START`; `START` with incoming edges |
| `_validate_duplicate_edges()` | Identical `(from, to)` pair appearing twice |
| `_validate_default_routes()` | Multiple `DEFAULT_ROUTE` edges from the same node |
| `_detect_unconditional_cycles()` | Cycles consisting entirely of unconditional (`route=None`) edges — these would loop forever |
| `_validate_static_schemas()` | Edge where `from_node.output_schema != to_node.input_schema` |

Conditional cycles (cycles with at least one `route=` edge) are **allowed** — they are how you build loop patterns.

After validation, `_compute_terminal_nodes()` populates `_terminal_node_names`:

```python
def _compute_terminal_nodes(self) -> None:
    from_names = {edge.from_node.name for edge in self.edges}
    self._terminal_node_names = {
        n.name for n in self.nodes
        if n.name != START.name and n.name not in from_names
    }
```

Terminal nodes (no outgoing edges) cause `Workflow._has_terminal_output()` to mark the workflow's output as delegated, preventing a duplicate output event.

### `get_next_pending_nodes()` — routing at runtime

```python
def get_next_pending_nodes(
    self, node_name: str, routes_to_match: RouteValue | list[RouteValue] | None
) -> list[str]:
```

Called by the workflow orchestration loop after each node completes. `routes_to_match` is whatever the node emitted via `ctx.emit_route()`. Returns the names of nodes whose triggers should be fired next, applying `DEFAULT_ROUTE` fallback logic.

### Complete example — routing map + default

```python
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._graph import Edge, DEFAULT_ROUTE
from google.adk.workflow._node import START
from google.adk.workflow._function_node import FunctionNode
from google.adk.agents.context import Context

async def classify(ctx: Context) -> None:
    text = ctx.node_input or ""
    if "urgent" in text.lower():
        ctx.emit_route("urgent")
    elif "billing" in text.lower():
        ctx.emit_route("billing")
    # No emit → DEFAULT_ROUTE fires

async def handle_urgent(ctx: Context) -> None:
    ctx.output = "Escalating to on-call team…"

async def handle_billing(ctx: Context) -> None:
    ctx.output = "Routing to billing department…"

async def handle_default(ctx: Context) -> None:
    ctx.output = "Handled by general support."

classifier    = FunctionNode(name="classify",        fn=classify)
urgent_node   = FunctionNode(name="handle_urgent",   fn=handle_urgent)
billing_node  = FunctionNode(name="handle_billing",  fn=handle_billing)
default_node  = FunctionNode(name="handle_default",  fn=handle_default)

workflow = Workflow(
    name="triage_workflow",
    edges=[
        (START, classifier, {
            "urgent":       urgent_node,
            "billing":      billing_node,
            DEFAULT_ROUTE:  default_node,
        }),
    ],
)
```

---

## 2 · `NodeRunner`

**Source:** `google.adk.workflow._node_runner`

`NodeRunner` is the per-node execution engine inside `Workflow`. Each time a node is scheduled, `Workflow._start_node_task()` creates a `NodeRunner` and wraps `runner.run()` in an asyncio Task. `NodeRunner` is responsible for:

- Creating a child `Context` (with sub-branch, isolation scope, and run ID)
- Iterating `node.run()` and enqueuing events to the invocation context event queue
- Flushing the final output event
- Detecting retryable failures and sleeping between attempts
- Carrying forward `prior_output` and `prior_interrupt_ids` on HITL resume

### Constructor (source-verified)

```python
class NodeRunner:
    def __init__(
        self,
        *,
        node: BaseNode,
        parent_ctx: Context,
        run_id: str | None = None,
        use_as_output: bool = False,
        prior_output: Any = None,
        prior_interrupt_ids: set[str] | None = None,
        use_sub_branch: bool = False,
        override_branch: str | None = None,
        override_isolation_scope: str | None = None,
    ) -> None:
```

| Parameter | Purpose |
|---|---|
| `run_id` | Sequential counter string (`"1"`, `"2"`, …); used to build the node's event branch path. Falls back to `"1"` if not provided. |
| `use_as_output` | When `True`, this node's output also counts as the **parent** node's output (used for `use_as_output=True` edges in `JoinNode`). |
| `prior_output` | Output carried forward from a previous run in a HITL resume scenario — pre-populates `ctx._output_value`. |
| `prior_interrupt_ids` | Unresolved interrupt IDs from the previous run; pre-populates `ctx._interrupt_ids`. |
| `use_sub_branch` | When `True`, appends `<name>@<run_id>` to the branch path (used for parallel branches and `_ParallelWorker` sub-runs). |
| `override_isolation_scope` | Explicitly overrides the session isolation scope (used for `mode='task'` LlmAgent nodes). |

### `run()` — the main execution loop

```python
async def run(
    self,
    node_input: Any = None,
    *,
    resume_inputs: dict[str, Any] | None = None,
) -> Context:
    attempt_count = 1
    while True:
        ctx = self._create_child_context(resume_inputs, attempt_count)
        try:
            async with node_tracing.start_as_current_node_span(...):
                await self._execute_node(ctx, node_input)
                await self._flush_output_and_deltas(ctx)
                return ctx
        except DynamicNodeFailError as e:
            ctx._error = e.error
            ctx._error_node_path = e.error_node_path
            return ctx
        except Exception as e:
            error_event = Event(error_code=type(e).__name__, error_message=str(e))
            await self._enqueue_event(error_event, ctx)
            if not await self._attempt_retry(e, ctx, attempt_count):
                ctx._error = e
                ctx._error_node_path = ctx.node_path
                return ctx
            attempt_count += 1
```

Retry state (`attempt_count`) is **in-memory only** — it is not persisted to the checkpoint. If a node is interrupted (HITL) mid-retry, the retry counter resets on the next resume.

### `_create_child_context()` — branch and scope setup

```python
def _create_child_context(
    self, resume_inputs: dict[str, Any] | None, attempt_count: int = 1
) -> Context:
```

Branch computation order:
1. If `override_branch` is set, use that directly.
2. If `use_sub_branch`, append `<name>@<run_id>` to the parent's branch (e.g. `main.worker@3`).
3. Otherwise inherit parent's branch unchanged.

If `override_isolation_scope` is set, the child context's `isolation_scope` is overridden — this is how `task`-mode `LlmAgent` nodes get a private conversation view.

### `_attempt_retry()` — exponential backoff gate

```python
async def _attempt_retry(
    self, e: Exception, ctx: Context, attempt_count: int
) -> bool:
    from .utils._retry_utils import _should_retry_node, _get_retry_delay
    node_state = NodeState(attempt_count=attempt_count)
    if not _should_retry_node(e, self._node.retry_config, node_state):
        return False
    delay = _get_retry_delay(self._node.retry_config, node_state)
    await asyncio.sleep(delay)
    return True
```

`_should_retry_node` checks `retry_config.exceptions` (if `None`, all exceptions are retried) and compares `attempt_count` against `retry_config.max_attempts` (default: 5). `_get_retry_delay` applies the exponential backoff formula with jitter.

### Observing NodeRunner behaviour — using OpenTelemetry spans

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

# Each node run creates a span named after the node.
# Wrap your runner usage with span inspection if you need
# per-node latency or failure counts:
async def run_with_tracing(runner, node_input):
    ctx = await runner.run(node_input)
    if ctx.error:
        print(f"Node failed: {ctx.error_node_path} — {ctx.error}")
    else:
        print(f"Node succeeded, output: {ctx.output}")
    return ctx
```

---

## 3 · `NodeState` + `NodeStatus`

**Source:** `google.adk.workflow._node_state`, `google.adk.workflow._node_status`

`NodeState` is the mutable per-node execution record that lives inside `Workflow._LoopState.nodes`. `NodeStatus` is the 7-value enum that tracks where in its lifecycle a node currently sits.

### `NodeStatus` — the 7-state lifecycle

```python
class NodeStatus(Enum):
    INACTIVE  = 0   # Not ready to execute
    PENDING   = 1   # Ready to execute (trigger received)
    RUNNING   = 2   # Currently executing
    COMPLETED = 3   # Finished successfully
    WAITING   = 4   # Paused (HITL interrupt or waiting for re-trigger)
    FAILED    = 5   # Execution failed
    CANCELLED = 6   # Cancelled (e.g. peer task failure in parallel branch)
```

State transitions driven by the workflow orchestration loop:

```
INACTIVE ──trigger──► RUNNING ──success──► COMPLETED
                          │
                          ├──interrupt──► WAITING ──resume──► RUNNING
                          │
                          └──exception──► FAILED
```

`WAITING` has two distinct sub-meanings (distinguished by `node_state.interrupts`):
- **Non-empty `interrupts`** — node is paused waiting for user HITL input. The scheduler skips new triggers for this node until the interrupts are resolved.
- **Empty `interrupts`** (but `status == WAITING`) — node produced no output yet but is re-triggerable to accumulate state (e.g. a `JoinNode` barrier waiting for all predecessors).

### `NodeState` — the execution record

```python
class NodeState(BaseModel):
    model_config = ConfigDict(extra='ignore', ser_json_bytes='base64')

    status:        NodeStatus         = NodeStatus.INACTIVE
    input:         Any                = None
    attempt_count: int                = Field(default=1, exclude_if=lambda v: v == 1)
    interrupts:    list[str]          = Field(default_factory=list)
    resume_inputs: dict[str, Any]     = Field(default_factory=dict)
    run_counter:   int                = Field(default=0, exclude_if=lambda v: v == 0)
    run_id:        str | None         = None
    parent_run_id: str | None         = None
```

Key field semantics:

| Field | Purpose |
|---|---|
| `attempt_count` | 1-based retry attempt counter. Fed into `_get_retry_delay` to compute exponential backoff sleep. Excluded from serialisation when `== 1` to minimise checkpoint size. |
| `interrupts` | List of pending interrupt IDs (from `ctx.interrupt()`). Non-empty → node is in HITL WAITING state. |
| `resume_inputs` | Dict keyed by interrupt ID → user-provided response. Populated when the runner resumes a WAITING node. |
| `run_counter` | Monotonically increasing counter, incremented each time a fresh `run_id` is assigned. Preserved across `NodeState` recreations to prevent path collisions between custom string IDs and auto-generated numeric IDs. |
| `run_id` | The current run's branch suffix (`"1"`, `"2"`, …). Combined with the node name to form the branch path `<name>@<run_id>`. |
| `parent_run_id` | The run ID of the parent node that dynamically scheduled this node via `ctx.run_node()`. Used for correlation in telemetry and replay. |

### Reading NodeState in tests

```python
import asyncio
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._node_status import NodeStatus
from google.adk.runners import InMemoryRunner
from google.adk.agents import LlmAgent

async def inspect_workflow_state():
    wf = Workflow(name="demo", edges=[...])
    runner = InMemoryRunner(agent=wf)
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1"
    )
    # Run one turn
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.session_id,
        new_message=types.Content(parts=[types.Part.from_text("go")]),
    ):
        # NodeState is not directly exposed on events, but the event's
        # author and branch fields reflect the NodeRunner's child context
        print(f"event from: {event.author}, branch: {event.branch}")

asyncio.run(inspect_workflow_state())
```

### `exclude_if` on serialisation

`NodeState` uses Pydantic's `Field(exclude_if=...)` to omit default-value fields from checkpoint serialisation. This means a `NodeState` for a node that completed on its first attempt without interrupts serialises to just `{"status": 3}` — keeping checkpoint size minimal.

---

## 4 · `_ParallelWorker`

**Source:** `google.adk.workflow._parallel_worker`

`_ParallelWorker` is the workflow node that implements **fan-out** parallel execution: given a list as `node_input`, it runs the wrapped node once per list item, concurrently, and yields a result list in the same order as the input. The `Workflow` class exposes this via the `parallel_worker()` helper.

### Constructor (source-verified)

```python
class _ParallelWorker(BaseNode):
    max_concurrency: int | None = Field(default=None)

    def __init__(
        self,
        *,
        node: NodeLike,      # Any node — FunctionNode, LlmAgent, Workflow, etc.
        max_concurrency: int | None = None,
        retry_config: RetryConfig | None = None,
        timeout: float | None = None,
    ):
```

- `node` is built via `build_node(node)` — you can pass a `FunctionNode`, `LlmAgent`, or another `Workflow`.
- `node == 'START'` is explicitly rejected (raises `ValueError`).
- The `_ParallelWorker`'s own name is set to the wrapped node's name.

### `_run_impl` — the fan-out loop

```python
async def _run_impl(self, *, ctx: Context, node_input: Any) -> AsyncGenerator[Any, None]:
    if not isinstance(node_input, list):
        node_input = [node_input]   # single-item list fallback
    if not node_input:
        yield []
        return

    results = [None] * len(node_input)
    pending_tasks: set[asyncio.Task] = set()
    input_index = 0

    while input_index < len(node_input) or pending_tasks:
        # Fill up to max_concurrency slots
        while input_index < len(node_input) and (
            self.max_concurrency is None
            or len(pending_tasks) < self.max_concurrency
        ):
            task = asyncio.create_task(
                ctx.run_node(self._node, node_input=node_input[input_index], use_sub_branch=True)
            )
            task._worker_index = input_index
            pending_tasks.add(task)
            input_index += 1

        done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if exc := task.exception():
                for p in pending_tasks:
                    p.cancel()
                await asyncio.wait(pending_tasks)
                raise exc
            results[task._worker_index] = task.result()

    yield results
```

Key behaviours:
- Uses `ctx.run_node(self._node, use_sub_branch=True)` — each item runs in a numbered sub-branch (e.g. `worker@1`, `worker@2`, …) so their events don't collide.
- Results are placed by index, not by completion order, so the output list always matches input order even when tasks finish out of order.
- If **any** task raises, all remaining tasks are cancelled immediately and the exception propagates to the workflow loop.

### Usage example — parallel document summariser

```python
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._parallel_worker import _ParallelWorker
from google.adk.workflow._function_node import FunctionNode
from google.adk.workflow._node import START
from google.adk.agents.context import Context
from google.adk.agents import LlmAgent

summariser = LlmAgent(
    name="summariser",
    model="gemini-2.5-flash",
    instruction="Summarise the document passed to you in one sentence.",
)

async def split_docs(ctx: Context) -> None:
    docs = ctx.node_input  # e.g. ["doc1 text...", "doc2 text..."]
    ctx.output = docs       # pass list to _ParallelWorker

async def collect(ctx: Context) -> None:
    summaries = ctx.node_input   # list[str] from _ParallelWorker
    ctx.output = "\n".join(f"- {s}" for s in summaries)

splitter  = FunctionNode(name="split_docs", fn=split_docs)
worker    = _ParallelWorker(node=summariser, max_concurrency=4)
collector = FunctionNode(name="collect", fn=collect)

pipeline = Workflow(
    name="doc_summariser",
    edges=[(START, splitter, worker, collector)],
)
```

> **Concurrency note:** `max_concurrency` limits the number of simultaneous `ctx.run_node()` calls. Since `_ParallelWorker` uses `FIRST_COMPLETED`, it eagerly starts new items as slots open — it does not wait for the entire current batch to finish before filling the next slot.

---

## 5 · `ActiveStreamingTool` + `TranscriptionEntry`

**Source:** `google.adk.agents.active_streaming_tool`, `google.adk.agents.transcription_entry`

These two small classes are the runtime state holders for live (bidirectional streaming) agent turns. They appear in `InvocationContext` and are managed by the live agent infrastructure — understanding them is essential when building custom streaming tools or inspecting live session state.

### `ActiveStreamingTool`

```python
class ActiveStreamingTool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    task:   Optional[asyncio.Task]     = None
    stream: Optional[LiveRequestQueue] = None
```

One `ActiveStreamingTool` instance exists per **active tool invocation** during a live turn. The live agent infrastructure stores these in `InvocationContext.active_streaming_tools` (a `dict[str, ActiveStreamingTool]` keyed by function call ID).

| Field | Purpose |
|---|---|
| `task` | The asyncio `Task` running the tool's `run_live()` coroutine. When the user interrupts or the turn ends, this task is cancelled. |
| `stream` | A `LiveRequestQueue` that the model uses to send new inputs **into the running tool** (e.g. new audio chunks while the tool is still processing the first ones). |

The `task` and `stream` fields hold live asyncio objects, which is why `arbitrary_types_allowed=True` is required.

### Lifecycle of a streaming tool

```
User speaks ──► model detects tool call ──► ActiveStreamingTool created
    │                                              │
    │             task = asyncio.create_task(     │
    │               tool.run_live(stream=stream)  │
    │             )                               │
    ▼                                             ▼
More audio ──────────────────────────────► stream.put(chunk)
    │
    ▼
User stops / model decides tool is done ──► task.cancel() or task completes
```

### `TranscriptionEntry`

```python
class TranscriptionEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    role: Optional[str] = None   # "user", "model", or None (for function calls)
    data: Union[types.Blob, types.Content]
```

`TranscriptionEntry` objects are accumulated in `InvocationContext.transcription_cache` during a live turn. After the turn ends, the transcription pipeline converts the raw audio blobs and model content into a text transcript for the session history.

| Field | Value |
|---|---|
| `role` | `"user"` for microphone audio; `"model"` for synthesised speech; `None` for function call/response parts |
| `data` | Either a raw `types.Blob` (audio bytes with `mime_type="audio/pcm"`) or a `types.Content` (already-transcribed text parts) |

### Building a custom streaming tool

```python
import asyncio
from google.adk.tools.base_tool import BaseTool
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.tools.tool_context import ToolContext
from google.adk import types

class EchoStreamingTool(BaseTool):
    """Echoes audio back to the model with a 100 ms delay."""

    def __init__(self):
        super().__init__(name="echo_audio", description="Echo audio with delay.")

    async def run_live(
        self,
        *,
        tool_context: ToolContext,
        stream: LiveRequestQueue,
    ):
        while True:
            request = await stream.get()
            if request is None:
                break   # stream closed
            await asyncio.sleep(0.1)
            # Send the blob back into the live session
            tool_context.send_audio(request.realtime_input.media_chunks[0])
```

When the live agent calls this tool, ADK creates `ActiveStreamingTool(task=<running coroutine>, stream=<LiveRequestQueue>)` and stores it in the invocation context. Audio chunks arriving from the user are routed to `stream` automatically.

---

## 6 · `CachePerformanceAnalyzer`

**Source:** `google.adk.utils.cache_performance_analyzer`  
**Status:** `@experimental`

`CachePerformanceAnalyzer` reads the event history from a completed session and produces a 13-metric performance report for a named agent's context-cache usage. It is the production tool for answering "is my context cache saving tokens, and by how much?"

### Constructor

```python
from google.adk.utils.cache_performance_analyzer import CachePerformanceAnalyzer
from google.adk.sessions import InMemorySessionService

analyzer = CachePerformanceAnalyzer(session_service=InMemorySessionService())
```

Takes any `BaseSessionService` — works with `InMemorySessionService`, `SqliteSessionService`, `FirestoreSessionService`, `VertexAiSessionService`.

### `analyze_agent_cache_performance()`

```python
report = await analyzer.analyze_agent_cache_performance(
    session_id="sess-abc",
    user_id="u1",
    app_name="my_app",
    agent_name="research_agent",
)
```

Returns a dict with two shapes:

**No cache data found:**
```python
{"status": "no_cache_data"}
```

**Cache data present:**

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `"active"` |
| `requests_with_cache` | `int` | Number of requests that used a context cache object |
| `avg_invocations_used` | `float` | Average number of times each cache object was reused |
| `latest_cache` | `str \| None` | Resource name of the most recently used cache |
| `cache_refreshes` | `int` | Number of distinct cache resource names (each refresh = new cache object) |
| `total_invocations` | `int` | Sum of `invocations_used` across all cache history entries |
| `total_prompt_tokens` | `int` | Total prompt tokens across all requests by this agent |
| `total_cached_tokens` | `int` | Total cached-content tokens across all requests |
| `cache_hit_ratio_percent` | `float` | `(cached_tokens / prompt_tokens) * 100` |
| `cache_utilization_ratio_percent` | `float` | `(requests_with_hits / total_requests) * 100` |
| `avg_cached_tokens_per_request` | `float` | Average cached tokens per request |
| `total_requests` | `int` | Total requests made by this agent in the session |
| `requests_with_cache_hits` | `int` | Number of requests where at least one token was served from cache |

### Data source

`CachePerformanceAnalyzer` reads from `event.cache_metadata` (a `CacheMetadata` object on each `Event`) and `event.usage_metadata.cached_content_token_count`. Cache metadata is populated automatically when `ContextCacheConfig` is attached to your `App`.

### Full production usage example

```python
import asyncio
from google.adk.apps.app import App
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.utils.cache_performance_analyzer import CachePerformanceAnalyzer

LARGE_SYSTEM_PROMPT = "..." * 500   # > 4096 tokens

agent = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction=LARGE_SYSTEM_PROMPT,
)

session_svc = InMemorySessionService()

app = App(
    agent=agent,
    session_service=session_svc,
    context_cache_config=ContextCacheConfig(
        cache_intervals=[0, 5, 10],   # cache at turns 0, 5, 10
        ttl_seconds=3600,
    ),
)

runner = Runner(app=app, session_service=session_svc)

async def main():
    session = await session_svc.create_session(app_name="demo", user_id="u1")
    for msg in ["Tell me about LLMs.", "What is RAG?", "Explain embeddings."]:
        async for _ in runner.run_async(
            user_id="u1", session_id=session.session_id,
            new_message=types.Content(parts=[types.Part.from_text(msg)]),
        ):
            pass

    analyzer = CachePerformanceAnalyzer(session_service=session_svc)
    report = await analyzer.analyze_agent_cache_performance(
        session_id=session.session_id,
        user_id="u1",
        app_name="demo",
        agent_name="researcher",
    )
    if report["status"] == "active":
        print(f"Cache hit ratio:  {report['cache_hit_ratio_percent']:.1f}%")
        print(f"Cache util ratio: {report['cache_utilization_ratio_percent']:.1f}%")
        print(f"Tokens saved:     {report['total_cached_tokens']}")
    else:
        print("No cache data recorded — check ContextCacheConfig min_tokens threshold.")

asyncio.run(main())
```

### `_get_agent_cache_history()` — filtering by agent

The private method `_get_agent_cache_history()` accepts an optional `agent_name=None` to retrieve cache metadata across **all** agents in a session. Pass `agent_name=None` when analysing multi-agent sessions to get a global view:

```python
all_cache_history = await analyzer._get_agent_cache_history(
    session_id=session_id, user_id="u1", app_name="app", agent_name=None
)
print(f"Total cache events across all agents: {len(all_cache_history)}")
```

---

## 7 · `StreamingResponseAggregator`

**Source:** `google.adk.utils.streaming_utils`

`StreamingResponseAggregator` is the engine that converts a stream of partial `LlmResponse` chunks — as they arrive from the Gemini Live API or SSE stream — into complete, ordered `types.Content` objects. It handles three interleaved streams simultaneously: text, thinking/thought parts, and function call arguments.

### Internal state

```python
class StreamingResponseAggregator:
    def __init__(self) -> None:
        self._text = ''
        self._thought_text = ''
        self._usage_metadata = None
        self._grounding_metadata = None
        self._citation_metadata = None
        self._response = None

        # Progressive SSE — ordered part accumulation
        self._parts_sequence: list[types.Part] = []
        self._current_text_buffer: str = ''
        self._current_text_is_thought: Optional[bool] = None
        self._finish_reason: Optional[types.FinishReason] = None

        # Streaming function call state
        self._current_fc_name: Optional[str] = None
        self._current_fc_args: dict[str, Any] = {}
        self._current_fc_id: Optional[str] = None
        self._current_thought_signature: Optional[bytes] = None
```

### Streaming function call accumulation

Function call arguments arrive as `PartialArg` objects with a `json_path` and a typed value (`string_value`, `number_value`, `bool_value`, `null_value`). The aggregator:

1. Uses `_get_value_from_partial_arg()` to extract the typed value; for strings, **appends** the chunk to any existing string at that path (enabling token-by-token streaming of string arguments).
2. Uses `_set_value_by_json_path()` to write the value into `_current_fc_args` using JSONPath notation (e.g. `$.location.latitude`).
3. When `fc.will_continue == False`, calls `_flush_function_call_to_sequence()` to emit a complete `FunctionCall` part.

### Text buffer flushing

The aggregator maintains a `_current_text_buffer` to merge consecutive text chunks of the same type (thought vs non-thought) before adding them to `_parts_sequence`. This prevents fragmented single-character `Part` objects:

```python
def _flush_text_buffer_to_sequence(self) -> None:
    if self._current_text_buffer:
        if self._current_text_is_thought:
            self._parts_sequence.append(
                types.Part(text=self._current_text_buffer, thought=True)
            )
        else:
            self._parts_sequence.append(
                types.Part.from_text(text=self._current_text_buffer)
            )
        self._current_text_buffer = ''
        self._current_text_is_thought = None
```

When the current chunk switches from text to function call (or vice versa), the buffer is flushed first, ensuring proper part ordering in the final `Content`.

### Practical use — building a custom streaming display

You won't normally instantiate `StreamingResponseAggregator` directly (ADK does it internally in `SSE` and `BIDI` streaming modes). But understanding its output is essential for building typewriter-effect UIs:

```python
from google.adk.runners import InMemoryRunner
from google.adk import types

runner = InMemoryRunner(agent=my_agent)
session = await runner.session_service.create_session(
    app_name="demo", user_id="u1"
)

import sys

async for event in runner.run_async(
    user_id="u1",
    session_id=session.session_id,
    new_message=types.Content(parts=[types.Part.from_text("Explain gravity.")]),
    run_config=RunConfig(streaming_mode=StreamingMode.SSE),
):
    if event.partial and event.content:
        # StreamingResponseAggregator emits partial=True events for typewriter
        for part in event.content.parts:
            if part.text and not part.thought:
                sys.stdout.write(part.text)
                sys.stdout.flush()
    elif not event.partial and event.content:
        # Final aggregated event — contains the complete text
        # Skip if you're already displaying partial chunks to avoid duplicate output
        pass
```

> **Avoiding duplicate text:** With `StreamingMode.SSE`, you receive both partial text events AND a final aggregated event containing the full text. Display only `event.partial == True` events for typewriter effect; ignore the final aggregated event's text, OR ignore partial events and only display the final.

### Thought signature preservation

When streaming thinking-enabled models (e.g. `gemini-2.5-flash`), the aggregator stores `_current_thought_signature` (a `bytes` value) and attaches it to the completed function call part via `fc_part.thought_signature = self._current_thought_signature`. This allows verifiable thought attribution in the final content.

---

## 8 · `AgentRefConfig` + `ArgumentConfig` + `CodeConfig`

**Source:** `google.adk.agents.common_configs`  
**Status:** `@experimental` (all three classes, via `FeatureName.AGENT_CONFIG`)

These three classes implement ADK's **YAML-based agent configuration DSL** — they are the building blocks that allow you to define entire multi-agent systems in YAML without writing Python. They are used by `App` and `LlmAgent` when loading configs from `.yaml` files.

### `AgentRefConfig` — referencing sub-agents

```python
class AgentRefConfig(BaseModel):
    config_path: Optional[str] = None   # mutually exclusive
    code:        Optional[str] = None   # mutually exclusive
```

Exactly one of `config_path` or `code` must be set (validated by `model_validator`):

| Field | Value | Meaning |
|---|---|---|
| `config_path` | `"search_agent.yaml"` | Load a sub-agent from a YAML file relative to the parent config's directory |
| `code` | `"my_lib.agents.my_agent"` | Import `my_agent` from `my_lib.agents` at runtime |

YAML usage:
```yaml
# coordinator.yaml
name: coordinator
model: gemini-2.5-pro
sub_agents:
  - config_path: search_agent.yaml
  - config_path: math_agent.yaml
  - code: my_project.specialist_agents.code_agent
```

### `ArgumentConfig` — typed constructor arguments

```python
class ArgumentConfig(BaseModel):
    name:  Optional[str] = None   # None for positional args
    value: Any
```

Used inside `CodeConfig.args` to pass constructor arguments to tools, callbacks, or any Python callable referenced by name.

### `CodeConfig` — referencing any Python callable

```python
class CodeConfig(BaseModel):
    name: str             # import path, e.g. "my_lib.tools.my_tool"
    args: Optional[List[ArgumentConfig]] = None
```

`CodeConfig` is how YAML config files reference tools (ADK built-ins or custom) and callbacks. `name` is resolved at runtime via `importlib`.

YAML usage for tools:
```yaml
# my_agent.yaml
name: my_agent
model: gemini-2.5-flash
instruction: You are a helpful assistant.
tools:
  - name: google_search          # ADK built-in
  - name: AgentTool
    args:
      - name: agent
        value: search_agent.yaml
      - name: skip_summarization
        value: true
  - name: my_project.tools.fetch_weather
    args:
      - name: api_key
        value: "${WEATHER_API_KEY}"
```

YAML usage for callbacks:
```yaml
before_model_callback:
  name: my_project.callbacks.rate_limiter
  args:
    - name: max_calls_per_minute
      value: 60
```

### Loading a YAML-configured agent

```python
from google.adk.agents.agent_config import load_agent_from_config

# Load an agent defined entirely in YAML
agent = load_agent_from_config("coordinator.yaml")
```

Or via `App`:
```python
from google.adk.apps.app import App

app = App.from_config("app_config.yaml")
```

### Validation: `AgentRefConfig` mutual exclusion

The `model_validator` enforces exactly-one-of semantics:

```python
@model_validator(mode="after")
def validate_exactly_one_field(self) -> AgentRefConfig:
    code_provided = self.code is not None
    config_path_provided = self.config_path is not None
    if code_provided and config_path_provided:
        raise ValueError("Only one of `code` or `config_path` should be provided")
    if not code_provided and not config_path_provided:
        raise ValueError("Exactly one of `code` or `config_path` must be provided")
    return self
```

---

## 9 · `BaseAuthProvider` + `AuthProviderRegistry`

**Source:** `google.adk.auth.base_auth_provider`, `google.adk.auth.auth_provider_registry`  
**Status:** `@experimental` (via `FeatureName.PLUGGABLE_AUTH`)

Vol. 5 covered `AuthConfig` + `AuthHandler` (the request-time credential resolution flow). `BaseAuthProvider` and `AuthProviderRegistry` are the **extension point** that lets you plug custom credential retrieval logic into that flow — without modifying ADK internals.

### `BaseAuthProvider` — the extension ABC

```python
class BaseAuthProvider(ABC):
    @property
    def supported_auth_schemes(self) -> tuple[type[AuthScheme], ...]:
        return ()   # override to enable 1-param registration

    @abstractmethod
    async def get_auth_credential(
        self,
        auth_config: AuthConfig,
        context: CallbackContext,
    ) -> AuthCredential | None:
        """Return an AuthCredential, or None if unavailable."""
```

`get_auth_credential` is called by `AuthHandler` when it needs to resolve credentials for a tool request. Return `None` to signal that this provider cannot satisfy the request; `AuthHandler` will fall back to the next registered provider or prompt the user.

The `supported_auth_schemes` property enables single-argument `registry.register(provider)` — ADK reads `provider.supported_auth_schemes` and registers it for each scheme type automatically.

### `AuthProviderRegistry` — the provider store

```python
class AuthProviderRegistry:
    def __init__(self):
        self._providers: dict[type[AuthScheme], BaseAuthProvider] = {}

    def register(
        self,
        auth_scheme_type: type[AuthScheme],
        provider_instance: BaseAuthProvider,
    ) -> None:
        self._providers[auth_scheme_type] = provider_instance

    def get_provider(
        self, auth_scheme: AuthScheme | type[AuthScheme]
    ) -> BaseAuthProvider | None:
        if isinstance(auth_scheme, type):
            return self._providers.get(auth_scheme)
        return self._providers.get(type(auth_scheme))
```

`get_provider()` accepts both the scheme class and an instance — `get_provider(OAuthScheme())` and `get_provider(OAuthScheme)` both work.

### Implementing a custom auth provider

The canonical use case: fetching API keys from a secret manager rather than from environment variables.

```python
import asyncio
from google.cloud import secretmanager

from google.adk.auth.base_auth_provider import BaseAuthProvider
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, ApiKey
from google.adk.auth.auth_schemes import ApiKeyScheme
from google.adk.agents.callback_context import CallbackContext

class GCPSecretManagerAuthProvider(BaseAuthProvider):
    """Retrieves API keys from GCP Secret Manager."""

    # Enable 1-param registration: registry.register(OurProvider, GCPSecretManagerAuthProvider(project="…"))
    supported_auth_schemes = (ApiKeyScheme,)

    def __init__(self, project: str, secret_name: str):
        self._project = project
        self._secret_name = secret_name
        self._client = secretmanager.SecretManagerServiceAsyncClient()

    async def get_auth_credential(
        self,
        auth_config,
        context: CallbackContext,
    ) -> AuthCredential | None:
        try:
            name = f"projects/{self._project}/secrets/{self._secret_name}/versions/latest"
            response = await self._client.access_secret_version(name=name)
            api_key = response.payload.data.decode("utf-8").strip()
            return AuthCredential(
                auth_type=AuthCredentialTypes.API_KEY,
                api_key=ApiKey(api_key=api_key),
            )
        except Exception:
            return None   # fall back to next provider


# Registration
from google.adk.auth.auth_provider_registry import AuthProviderRegistry

registry = AuthProviderRegistry()
registry.register(
    ApiKeyScheme,
    GCPSecretManagerAuthProvider(project="my-gcp-project", secret_name="weather-api-key"),
)

# Attach to App
app = App(agent=agent, auth_provider_registry=registry)
```

### Provider lookup in `AuthHandler`

`AuthHandler._resolve_credential()` calls `registry.get_provider(auth_scheme)` before attempting other resolution strategies. This means your provider runs first — if it returns `None`, ADK falls through to environment-variable lookup and then to user prompting.

---

## 10 · `FinishTaskTool` + `TaskRequest` + `TaskResult` + `_DefaultTaskOutput`

**Source:** `google.adk.agents.llm.task._finish_task_tool`, `google.adk.agents.llm.task._task_models`

These four classes implement the **task delegation protocol** — the internal machinery that makes `LlmAgent(mode='task')` work. When a coordinator LlmAgent delegates work to a task sub-agent, `FinishTaskTool` is injected into the task agent's tool list, and `TaskRequest`/`TaskResult` are the typed wire-format for the delegation.

### `TaskRequest` + `TaskResult` — the wire format

```python
class TaskRequest(BaseModel):
    # camelCase aliases: agentName, input
    agent_name: str            # the target sub-agent's name
    input:      dict[str, Any] # validated input data

class TaskResult(BaseModel):
    # camelCase alias: output
    output: Any                # validated output data from the task agent
```

These Pydantic models use `alias_generator=alias_generators.to_camel` with `populate_by_name=True`, so both snake_case and camelCase field names are accepted.

### `_DefaultTaskOutput` — the fallback output schema

```python
class _DefaultTaskOutput(BaseModel):
    result: str   # "A brief summary of what the agent accomplished."
```

When a `mode='task'` `LlmAgent` is defined without an explicit `output_schema`, `FinishTaskTool` uses `_DefaultTaskOutput` as the schema for its `finish_task` function declaration. The model must call `finish_task(result="...")` to signal completion.

Similarly, `_DefaultTaskInput` is the fallback input schema (used by the coordinator's `RequestTaskTool`):

```python
class _DefaultTaskInput(BaseModel):
    goal:       Optional[str] = None   # the task objective
    background: Optional[str] = None   # additional context
```

### `FinishTaskTool` — the completion signal

```python
class FinishTaskTool(BaseTool):
    def __init__(self, task_agent: LlmAgent):
        output_schema = task_agent.output_schema or _DefaultTaskOutput
        self._adapter = TypeAdapter(output_schema)
        raw_schema = self._adapter.json_schema()
        # If schema is not a JSON object, wrap it: {"result": <schema>}
        self._wrapper_key = None if raw_schema.get('type') == 'object' else 'result'
```

`FinishTaskTool` is injected automatically into task agents by the `LlmAgent` when `mode='task'` is set — you do not construct it manually.

**What `FinishTaskTool` does at runtime:**

1. **`process_llm_request()`** — appends a system instruction to the LLM request:
   > *"Do NOT call `finish_task` prematurely. Use your available tools to fully complete every aspect of the delegated task first…"*

2. **`_get_declaration()`** — generates the `FunctionDeclaration` for `finish_task` from `output_schema`. If the schema is a primitive (not an object), it wraps it: `{"result": <schema>}`. `$defs` are hoisted to root level to keep `$ref` pointers valid.

3. **`run_async()`** — validates the LLM's arguments against `output_schema` using `TypeAdapter`. On validation failure, returns an error dict prompting the model to retry with corrected types. On success, returns `FINISH_TASK_SUCCESS_RESULT` — the coordinator's event loop detects this and sets `event.output` on the task agent's run.

### Defining a typed task agent

```python
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.tools.function_tool import FunctionTool
from google.adk import types

class ResearchOutput(BaseModel):
    summary: str
    sources: list[str]
    confidence: float

async def web_search(query: str) -> str:
    return f"Results for: {query}"

research_agent = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    mode="task",                       # task agent — FinishTaskTool injected automatically
    output_schema=ResearchOutput,      # _DefaultTaskOutput replaced by ResearchOutput
    input_schema=None,                 # uses _DefaultTaskInput (goal + background)
    instruction=(
        "Research the given goal thoroughly. "
        "Call finish_task with a structured summary, source list, and confidence score."
    ),
    tools=[FunctionTool(fn=web_search)],
)

coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.5-pro",
    instruction="Delegate research tasks to the researcher sub-agent.",
    sub_agents=[research_agent],
)
```

When the coordinator delegates to `researcher`:
1. ADK calls `RequestTaskTool` (on the coordinator) to create a `TaskRequest(agent_name="researcher", input={"goal": "…"})`.
2. The researcher runs with `FinishTaskTool` injected. The model calls `finish_task(summary="…", sources=[…], confidence=0.92)`.
3. `FinishTaskTool.run_async()` validates the call against `ResearchOutput`, returns success, and ADK stores the result as `TaskResult(output={"summary": "…", "sources": […], "confidence": 0.92})`.
4. The coordinator receives the `TaskResult` and continues its turn.

### Validation error feedback loop

When the model passes incorrect types to `finish_task`:

```python
# Model calls finish_task(summary=42, sources="not a list", confidence="high")
# FinishTaskTool.run_async() catches the ValidationError and returns:
{
    "error": (
        "Invoking `finish_task()` failed due to validation errors:\n"
        "3 validation errors for ResearchOutput\n"
        "  summary: str expected ...\n"
        "  sources: list expected ...\n"
        "  confidence: float expected ...\n"
        "You could retry calling this tool, but it is IMPORTANT for you to "
        "provide all the mandatory parameters with correct types."
    )
}
```

The error message is returned as the tool result, prompting the model to retry with corrected arguments on its next turn.

---

## Cross-references

- **Workflow internals:** [Vol. 1 — `Workflow`, `RetryConfig`, `BaseNode`](./google_adk_class_deep_dives/) · [Vol. 8 — `FunctionNode`, `JoinNode`, `Trigger`](./google_adk_class_deep_dives_v8/) · [Vol. 7 — `DynamicNodeScheduler`, `ctx.run_node()`](./google_adk_class_deep_dives_v7/)
- **Auth:** [Vol. 5 — `AuthConfig`, `AuthHandler`](./google_adk_class_deep_dives_v5/) · [Vol. 2 — `AuthCredential`](./google_adk_class_deep_dives_v2/)
- **Streaming:** [Vol. 4 — `LiveRequestQueue`, `LiveRequest`](./google_adk_class_deep_dives_v4/)
- **Context caching:** [Vol. 9 — `ContextCacheConfig`, `GeminiContextCacheManager`](./google_adk_class_deep_dives_v9/)
- **Task agents:** [Vol. 4 — `LongRunningFunctionTool`](./google_adk_class_deep_dives_v4/) · [Vol. 8 — `ReadonlyContext`](./google_adk_class_deep_dives_v8/)
- **YAML config:** [Comprehensive guide — `App`, `LlmAgent`](./google_adk_comprehensive_guide/)

---
title: "Class deep dives — volume 26 (LangGraphAgent, PubSubToolset, SpannerToolset/SpannerVectorStoreSettings, LongRunningFunctionTool, ContextCacheConfig, LlmEventSummarizer, ToolConfirmation, ToolboxToolset, DynamicNodeScheduler, FileArtifactService)"
description: "Source-verified deep dives into 10 google-adk 2.3.0 classes: LangGraphAgent (checkpointer-aware message routing; multi-turn graph state; sub-agent composition), PubSubToolset (publish/pull/acknowledge; ordering_key; auto-ack; PubSubToolConfig; @experimental), SpannerToolset+SpannerToolSettings+SpannerVectorStoreSettings (QueryResultMode.DICT_LIST; get_execute_sql factory; ANN vector search; SpannerVectorStoreSettings full field reference), LongRunningFunctionTool (is_long_running flag; modified _get_declaration; polling pattern), ContextCacheConfig (cache_intervals/ttl_seconds/min_tokens; create_http_options timeout; 4096-token floor; second-turn-start; @experimental), LlmEventSummarizer (_DEFAULT_PROMPT_TEMPLATE; _MAX_TOOL_CONTENT_CHARS=2000; thought + FC + FR formatting; EventCompaction output), ToolConfirmation (hint/confirmed/payload; camelCase aliases; HITL tool approval gate; @experimental), ToolboxToolset (MCP Toolbox delegate; toolset_name + tool_names union; bound_params; auth_token_getters; lazy import guard), DynamicNodeScheduler+DynamicNodeRun+DynamicNodeState (3-case dedup/resume/fresh algorithm; rehydration from session events; HITL interrupt propagation; ReplaySequenceBarrier), FileArtifactService (versioned disk layout; user: namespace; path-traversal guard; session vs user scope; metadata.json sidecar)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 26"
  order: 95
---

Source-verified against **google-adk==2.3.0** (installed from PyPI, June 2026). Every field name, signature, constant, and code example is drawn directly from the installed package source.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `LangGraphAgent` | `google.adk.agents.langgraph_agent` | Stable |
| 2 | `PubSubToolset` + `publish_message` / `pull_messages` / `acknowledge_messages` | `google.adk.tools.pubsub.*` | `@experimental` |
| 3 | `SpannerToolset` + `SpannerToolSettings` + `SpannerVectorStoreSettings` | `google.adk.tools.spanner.*` | `@experimental` |
| 4 | `LongRunningFunctionTool` | `google.adk.tools.long_running_tool` | Stable |
| 5 | `ContextCacheConfig` | `google.adk.agents.context_cache_config` | `@experimental` |
| 6 | `LlmEventSummarizer` | `google.adk.apps.llm_event_summarizer` | Stable |
| 7 | `ToolConfirmation` | `google.adk.tools.tool_confirmation` | `@experimental` |
| 8 | `ToolboxToolset` | `google.adk.tools.toolbox_toolset` | Stable |
| 9 | `DynamicNodeScheduler` + `DynamicNodeRun` + `DynamicNodeState` | `google.adk.workflow._dynamic_node_scheduler` | Stable |
| 10 | `FileArtifactService` | `google.adk.artifacts.file_artifact_service` | Stable |

---

## 1 · `LangGraphAgent`

**Source:** `google.adk.agents.langgraph_agent`

`LangGraphAgent` is a `BaseAgent` subclass that wraps a LangGraph `CompiledGraph` so it can participate in an ADK multi-agent system. It bridges ADK's `Event`-based session history with LangGraph's `messages` state, and it respects LangGraph's own checkpointer if one is configured.

### Constructor (source-verified)

```python
class LangGraphAgent(BaseAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: CompiledGraph
    instruction: str = ""
```

| Field | Type | Notes |
|---|---|---|
| `graph` | `CompiledGraph` | Any LangGraph graph compiled with `.compile()`; optional `checkpointer` controls message routing |
| `instruction` | `str` | Injected as `SystemMessage` on the **first** turn only (when `graph.get_state()` returns empty) |

### Message routing logic (source-verified)

The key design decision is how ADK's event log maps to LangGraph messages:

```python
def _get_messages(self, events: list[Event]) -> list[...]:
    if self.graph.checkpointer:
        # LangGraph owns the full history; only send the latest user turn
        return _get_last_human_messages(events)
    else:
        # No checkpointer — feed the full ADK session as messages
        return self._get_conversation_with_agent(events)
```

- **With checkpointer** (`InMemorySaver`, `SqliteSaver`, etc.) — LangGraph stores the full graph state across turns; `LangGraphAgent` sends only the most recent user messages to avoid duplicating history. The checkpointer is keyed to `ctx.session.id` via `configurable.thread_id`.
- **Without checkpointer** — `LangGraphAgent` reconstructs the full ADK ↔ LangGraph conversation by scanning `events` for `event.author == 'user'` (→ `HumanMessage`) and `event.author == self.name` (→ `AIMessage`).

### `_run_async_impl` internals

```python
async def _run_async_impl(self, ctx: InvocationContext):
    config: RunnableConfig = {"configurable": {"thread_id": ctx.session.id}}

    # Only inject instruction on the very first turn
    current_graph_state = self.graph.get_state(config)
    graph_messages = (
        current_graph_state.values.get("messages", [])
        if current_graph_state.values
        else []
    )
    messages = (
        [SystemMessage(content=self.instruction)]
        if self.instruction and not graph_messages
        else []
    )

    messages += self._get_messages(ctx.session.events)
    final_state = self.graph.invoke({"messages": messages}, config)
    result = final_state["messages"][-1].content

    yield Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        branch=ctx.branch,
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=result)],
        ),
    )
```

### Example 1 — minimal single-turn graph agent

```python
import asyncio
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from google.adk.agents import LlmAgent
from google.adk.agents.langgraph_agent import LangGraphAgent
from google.adk.runners import InMemoryRunner

# 1. Build and compile the LangGraph graph
def call_model(state: MessagesState):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("llm", call_model)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)
graph = builder.compile()

# 2. Wrap as an ADK agent
lg_agent = LangGraphAgent(
    name="langgraph_agent",
    instruction="You are a helpful assistant. Be concise.",
    graph=graph,
)

# 3. Wrap in an ADK root agent for routing
root_agent = LlmAgent(
    name="root",
    model="gemini-2.0-flash",
    sub_agents=[lg_agent],
    instruction="Route all questions to langgraph_agent.",
)

async def main():
    runner = InMemoryRunner(agent=root_agent, app_name="lg_demo")
    session = await runner.session_service.create_session(
        app_name="lg_demo", user_id="u1"
    )
    async for event in runner.run_async(
        session_id=session.id,
        user_id="u1",
        new_message=types.Content(role="user", parts=[types.Part.from_text("What is 2+2?")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — multi-turn with `InMemorySaver` checkpointer

```python
from langgraph.checkpoint.memory import MemorySaver

# Compile with checkpointer for multi-turn memory
graph_with_memory = builder.compile(checkpointer=MemorySaver())

lg_agent = LangGraphAgent(
    name="stateful_agent",
    instruction="You are a helpful assistant that remembers context.",
    graph=graph_with_memory,
    # With checkpointer: LangGraphAgent only passes the latest user message;
    # LangGraph's own state holds the full conversation history.
)
```

### Example 3 — custom graph state schema

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str

def personalised_model(state: AgentState):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    name = state.get("user_name", "there")
    # Inject the user's name into context
    enriched = list(state["messages"])
    enriched[-1].content = f"[Speaking to {name}]: " + enriched[-1].content
    return {"messages": [llm.invoke(enriched)]}

builder2 = StateGraph(AgentState)
builder2.add_node("llm", personalised_model)
builder2.set_entry_point("llm")
builder2.set_finish_point("llm")
# NOTE: LangGraphAgent always passes {"messages": ...} as graph input;
# custom state keys like `user_name` must be pre-seeded via initial_state
# or set in the graph definition's default factory.
```

---

## 2 · `PubSubToolset` + pub/sub tools

**Source:** `google.adk.tools.pubsub.pubsub_toolset`, `.message_tool`, `.config`

`PubSubToolset` is an `@experimental` `BaseToolset` that surfaces three Pub/Sub operations — publish, pull, and acknowledge — as ADK tools. The tools are plain Python functions wrapped as `GoogleTool` instances, which handle GCP credential injection automatically.

### Constructor (source-verified)

```python
@experimental(FeatureName.PUBSUB_TOOLSET)
class PubSubToolset(BaseToolset):
    def __init__(
        self,
        *,
        tool_filter: ToolPredicate | list[str] | None = None,
        credentials_config: PubSubCredentialsConfig | None = None,
        pubsub_tool_config: PubSubToolConfig | None = None,
    ): ...
```

| Parameter | Purpose |
|---|---|
| `tool_filter` | `None` → all 3 tools; `list[str]` → include by name; `ToolPredicate` → callable `(tool, ctx) → bool` |
| `credentials_config` | GCP credential configuration (ADC, service account, etc.) |
| `pubsub_tool_config` | `PubSubToolConfig(project_id=...)` — sets the GCP project; inferred from environment if `None` |

### `PubSubToolConfig` (source-verified)

```python
@experimental(FeatureName.PUBSUB_TOOL_CONFIG)
class PubSubToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str | None = None
    # Inferred from environment or credentials if not set
```

### Tools provided (source-verified)

The toolset exposes exactly three tools:

| Tool name | Function | Purpose |
|---|---|---|
| `publish_message` | `message_tool.publish_message` | Publish a message to a Pub/Sub topic |
| `pull_messages` | `message_tool.pull_messages` | Pull messages from a subscription |
| `acknowledge_messages` | `message_tool.acknowledge_messages` | Acknowledge pulled messages |

### `publish_message` signature

```python
def publish_message(
    topic_name: str,          # e.g. "projects/my-project/topics/my-topic"
    message: str,             # UTF-8 message body
    credentials: Credentials, # injected by GoogleTool
    settings: PubSubToolConfig,
    attributes: Optional[dict[str, str]] = None,  # metadata key-value pairs
    ordering_key: str = "",   # enables ordered delivery on the publisher
) -> dict:
    # Returns {"message_id": "..."} on success
    # Returns {"status": "ERROR", "error_details": "..."} on failure
```

### `pull_messages` signature

```python
def pull_messages(
    subscription_name: str,   # e.g. "projects/my-project/subscriptions/my-sub"
    credentials: Credentials,
    settings: PubSubToolConfig,
    *,
    max_messages: int = 1,
    auto_ack: bool = False,   # if True, acknowledges immediately after pulling
) -> dict:
    # Returns {"messages": [...]} where each message has:
    # message_id, data (str), attributes (dict), ordering_key, publish_time, ack_id
```

Data decoding falls back to Base64 (`ascii`) if UTF-8 decoding fails — so binary Pub/Sub payloads are handled gracefully.

### `acknowledge_messages` signature

```python
def acknowledge_messages(
    subscription_name: str,
    ack_ids: list[str],       # ack_id values from pull_messages response
    credentials: Credentials,
    settings: PubSubToolConfig,
) -> dict:
    # Returns {"status": "SUCCESS"} or {"status": "ERROR", "error_details": "..."}
```

### Example 1 — basic event-driven agent

```python
from google.adk.agents import LlmAgent
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig

pubsub_toolset = PubSubToolset(
    pubsub_tool_config=PubSubToolConfig(project_id="my-gcp-project"),
)

agent = LlmAgent(
    name="pubsub_agent",
    model="gemini-2.0-flash",
    toolsets=[pubsub_toolset],
    instruction=(
        "You can publish and consume Pub/Sub messages. "
        "Pull at most 5 messages from a subscription and "
        "acknowledge them only after confirming their content."
    ),
)
```

### Example 2 — filtered toolset (publish-only)

```python
# Only expose publish_message; hide pull and acknowledge
publish_only = PubSubToolset(
    pubsub_tool_config=PubSubToolConfig(project_id="my-gcp-project"),
    tool_filter=["publish_message"],
)
```

### Example 3 — ordered delivery with `ordering_key`

```python
# The agent can pass ordering_key to guarantee ordered delivery per key.
# Enable message ordering on the publisher by setting ordering_key != "".
# The publisher client is automatically created with enable_message_ordering=True
# when ordering_key is non-empty (source-verified).
instruction = """
When publishing batch events, always use the same ordering_key value
for related events (e.g. the customer_id) so they arrive in order.
Topic: projects/my-project/topics/order-events
"""
```

### Example 4 — manual pull-then-ack pattern

```python
# Instruction to guide the agent through the two-step pull/acknowledge flow:
instruction = """
1. Pull up to 10 messages from projects/my-project/subscriptions/order-sub
   (set auto_ack=False so you can inspect before acknowledging).
2. Process each message.
3. Acknowledge all messages using their ack_ids.
"""
```

---

## 3 · `SpannerToolset` + `SpannerToolSettings` + `SpannerVectorStoreSettings`

**Source:** `google.adk.tools.spanner.spanner_toolset`, `.settings`, `.query_tool`

`SpannerToolset` exposes Cloud Spanner as a set of read-only ADK tools for metadata inspection, SQL querying, and vector similarity search. It is decorated with `@experimental`.

### Constructor (source-verified)

```python
@experimental(FeatureName.SPANNER_TOOLSET)
class SpannerToolset(BaseToolset):
    def __init__(
        self,
        *,
        tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
        credentials_config: Optional[SpannerCredentialsConfig] = None,
        spanner_tool_settings: Optional[SpannerToolSettings] = None,
    ): ...
```

### Tools provided (source-verified)

| Tool name (after `spanner_` prefix) | Source function | Capability gate |
|---|---|---|
| `spanner_list_table_names` | `metadata_tool.list_table_names` | always |
| `spanner_list_table_indexes` | `metadata_tool.list_table_indexes` | always |
| `spanner_list_table_index_columns` | `metadata_tool.list_table_index_columns` | always |
| `spanner_list_named_schemas` | `metadata_tool.list_named_schemas` | always |
| `spanner_get_table_schema` | `metadata_tool.get_table_schema` | always |
| `spanner_execute_sql` | `query_tool.execute_sql` / dict-list variant | `Capabilities.DATA_READ` |
| `spanner_similarity_search` | `search_tool.similarity_search` | `Capabilities.DATA_READ` |
| `spanner_vector_store_similarity_search` | `search_tool.vector_store_similarity_search` | `Capabilities.DATA_READ` + `vector_store_settings` present |

### `SpannerToolSettings` field reference (source-verified)

```python
@experimental(FeatureName.SPANNER_TOOL_SETTINGS)
class SpannerToolSettings(BaseModel):
    capabilities: List[Capabilities] = [Capabilities.DATA_READ]
    max_executed_query_result_rows: int = 50
    query_result_mode: QueryResultMode = QueryResultMode.DEFAULT
    database_role: Optional[str] = None
    vector_store_settings: Optional[SpannerVectorStoreSettings] = None
```

### `QueryResultMode` — two output formats

```python
class QueryResultMode(Enum):
    DEFAULT = "default"    # list of rows: [["val1", "val2"], ...]
    DICT_LIST = "dict_list"  # list of dicts: [{"col1": "val1", "col2": "val2"}, ...]
```

When `QueryResultMode.DICT_LIST` is set, `get_execute_sql()` returns a **closure** that has the same implementation but a different docstring — the docstring change teaches the LLM to expect dict-keyed rows, avoiding confusion when it has to reference specific columns by name.

### `SpannerVectorStoreSettings` field reference (source-verified)

```python
class SpannerVectorStoreSettings(BaseModel):
    project_id: str
    instance_id: str
    database_id: str
    table_name: str
    content_column: str        # text column returned in results
    embedding_column: str      # FLOAT64 ARRAY column
    vector_length: int         # must match embedding model output dimension
    vertex_ai_embedding_model_name: str  # e.g. "text-embedding-005"
    selected_columns: list[str] = []    # default: [content_column]
    nearest_neighbors_algorithm: Literal[
        "EXACT_NEAREST_NEIGHBORS", "APPROXIMATE_NEAREST_NEIGHBORS"
    ] = "EXACT_NEAREST_NEIGHBORS"
    top_k: int = 4
    distance_type: str = "COSINE"       # COSINE | DOT_PRODUCT | EUCLIDEAN
    num_leaves_to_search: Optional[int] = None  # ANN only
    additional_filter: Optional[str] = None     # added to WHERE clause
    vector_search_index_settings: Optional[VectorSearchIndexSettings] = None  # ANN only
    additional_columns_to_setup: Optional[list[TableColumn]] = None
    primary_key_columns: Optional[list[str]] = None
```

A `@model_validator` enforces that `vector_length > 0` and that any `primary_key_columns` are present in `additional_columns_to_setup`.

### `VectorSearchIndexSettings` field reference (source-verified)

```python
class VectorSearchIndexSettings(BaseModel):
    index_name: str
    additional_key_columns: Optional[list[str]] = None
    additional_storing_columns: Optional[list[str]] = None
    tree_depth: int = 2       # 2 or 3
    num_leaves: int = 1000    # recommended: num_rows / 1000
    num_branches: Optional[int] = None  # only for tree_depth=3
```

### Example 1 — read-only SQL agent

```python
from google.adk.agents import LlmAgent
from google.adk.tools.spanner.spanner_toolset import SpannerToolset
from google.adk.tools.spanner.settings import SpannerToolSettings, QueryResultMode

spanner_toolset = SpannerToolset(
    spanner_tool_settings=SpannerToolSettings(
        max_executed_query_result_rows=100,
        query_result_mode=QueryResultMode.DICT_LIST,  # column-named dicts
    ),
)

agent = LlmAgent(
    name="spanner_analyst",
    model="gemini-2.0-flash",
    toolsets=[spanner_toolset],
    instruction=(
        "You are a Spanner analyst. Project: my-project, "
        "Instance: my-instance, Database: my-db. "
        "Answer questions by running SQL queries."
    ),
)
```

### Example 2 — vector similarity search agent

```python
from google.adk.tools.spanner.settings import (
    SpannerToolSettings,
    SpannerVectorStoreSettings,
    VectorSearchIndexSettings,
)

vector_settings = SpannerVectorStoreSettings(
    project_id="my-project",
    instance_id="my-instance",
    database_id="my-db",
    table_name="documents",
    content_column="content",
    embedding_column="embedding",
    vector_length=768,  # must match text-embedding-005 output
    vertex_ai_embedding_model_name="text-embedding-005",
    selected_columns=["title", "content", "category"],
    top_k=5,
    distance_type="COSINE",
    nearest_neighbors_algorithm="APPROXIMATE_NEAREST_NEIGHBORS",
    num_leaves_to_search=50,
    vector_search_index_settings=VectorSearchIndexSettings(
        index_name="documents_embedding_idx",
        num_leaves=1000,
        tree_depth=2,
    ),
)

spanner_toolset = SpannerToolset(
    spanner_tool_settings=SpannerToolSettings(
        vector_store_settings=vector_settings,
    ),
)
```

### Example 3 — metadata-only (no SQL capability)

```python
from google.adk.tools.spanner.settings import SpannerToolSettings, Capabilities

# Remove DATA_READ capability: only metadata tools are exposed
metadata_only_toolset = SpannerToolset(
    spanner_tool_settings=SpannerToolSettings(
        capabilities=[],  # empty list removes execute_sql and similarity_search
    ),
    tool_filter=["spanner_list_table_names", "spanner_get_table_schema"],
)
```

---

## 4 · `LongRunningFunctionTool`

**Source:** `google.adk.tools.long_running_tool`

`LongRunningFunctionTool` is a `FunctionTool` subclass for async operations that take significant time. Setting `is_long_running = True` signals the framework to treat the tool's return as an *intermediate* async result, with the final answer delivered later via the function call ID.

### Constructor (source-verified)

```python
class LongRunningFunctionTool(FunctionTool):
    def __init__(self, func: Callable):
        super().__init__(func)
        self.is_long_running = True  # marks the tool for async handling
```

### `_get_declaration` override (source-verified)

The override appends a guardrail instruction to the tool's docstring that prevents the LLM from calling the tool a second time before the async result arrives:

```python
@override
def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    declaration = super()._get_declaration()
    if declaration:
        instruction = (
            "\n\nNOTE: This is a long-running operation. Do not call this tool"
            " again if it has already returned some intermediate or pending"
            " status."
        )
        if declaration.description:
            declaration.description += instruction
        else:
            declaration.description = instruction.lstrip()
    return declaration
```

### When to use

Use `LongRunningFunctionTool` when:
- The function starts an async job and immediately returns a pending status or job ID.
- The final result arrives later (polled or pushed via another mechanism).
- You want to prevent the LLM from redundantly re-calling the tool during the wait.

### Example 1 — basic polling pattern

```python
import asyncio
import uuid
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.agents import LlmAgent

# Simulated long-running job store
_jobs: dict[str, str] = {}

async def start_report_generation(report_type: str) -> dict:
    """Generate a business report. Returns a job_id immediately; poll for status."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = "pending"
    # Fire off background work (in production this would be a real async job)
    asyncio.create_task(_simulate_report(job_id, report_type))
    return {"job_id": job_id, "status": "pending"}

async def _simulate_report(job_id: str, report_type: str):
    await asyncio.sleep(3)
    _jobs[job_id] = f"completed: {report_type} report ready"

async def check_report_status(job_id: str) -> dict:
    """Check the status of a previously started report generation job."""
    status = _jobs.get(job_id, "not_found")
    return {"job_id": job_id, "status": status}

agent = LlmAgent(
    name="report_agent",
    model="gemini-2.0-flash",
    tools=[
        LongRunningFunctionTool(start_report_generation),  # async start
        check_report_status,                                # polling check
    ],
    instruction=(
        "Start report generation and then poll check_report_status "
        "until status is 'completed'. Report the final result."
    ),
)
```

### Example 2 — combining with `ToolConfirmation`

```python
from google.adk.tools.tool_confirmation import ToolConfirmation

async def deploy_to_production(service_name: str, version: str) -> dict | ToolConfirmation:
    """Deploy a service to production. Requires human confirmation first."""
    # Return ToolConfirmation to pause and request human approval
    return ToolConfirmation(
        hint=f"Confirm deployment of {service_name} v{version} to production?",
        confirmed=False,
        payload={"service": service_name, "version": version},
    )

deploy_tool = LongRunningFunctionTool(deploy_to_production)
```

---

## 5 · `ContextCacheConfig`

**Source:** `google.adk.agents.context_cache_config`

`ContextCacheConfig` configures Gemini's [context caching](https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview) feature for all `LlmAgent` instances within an `App`. When this object is present on the `App`, context caching is enabled app-wide. When absent (`None`), caching is disabled.

The class is decorated with `@experimental(FeatureName.AGENT_CONFIG)`.

### Field reference (source-verified)

```python
@experimental(FeatureName.AGENT_CONFIG)
class ContextCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_intervals: int = Field(default=10, ge=1, le=100)
    ttl_seconds: int = Field(default=1800, gt=0)        # 30 minutes
    min_tokens: int = Field(default=0, ge=0)
    create_http_options: types.HttpOptions | None = Field(default=None)
```

| Field | Default | Meaning |
|---|---|---|
| `cache_intervals` | `10` | Maximum number of consecutive invocations that reuse the same cache before it is refreshed. Range: 1–100. |
| `ttl_seconds` | `1800` | Time-to-live for the cache, in seconds. Cache is automatically invalidated after this period. |
| `min_tokens` | `0` | Minimum prior-request token count required to enable caching. Gemini's hard floor is **4096 tokens** — values below 4096 have no additional effect. |
| `create_http_options` | `None` | `types.HttpOptions` to control the `CachedContent.create()` API call (e.g. add a timeout). On timeout, the request proceeds without caching. |

### Important constraints (source-verified)

From the field description:

- **Second-turn minimum**: caching begins on the second turn of a session at the earliest. The first request has no prior token count to evaluate against `min_tokens`, so no cache is created.
- **4096-token hard floor**: Gemini always enforces this minimum regardless of `min_tokens`. Setting `min_tokens=4096` or higher adds an additional ADK-side gate.
- **Short sessions are never cached**: single-turn sessions or sessions with small context will never trigger caching even if `ContextCacheConfig` is present.

### Helper property (source-verified)

```python
@property
def ttl_string(self) -> str:
    return f"{self.ttl_seconds}s"  # e.g. "1800s" — used in CachedContent.create()
```

### Example 1 — enabling context caching on an app

```python
from google.adk.agents import LlmAgent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.genai import types as genai_types

agent = LlmAgent(
    name="assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant with access to a large document corpus.",
    static_instruction=genai_types.Content(
        parts=[genai_types.Part.from_text(LARGE_DOCUMENT_TEXT)],
    ),
)

# Enable context caching: cache refreshes every 5 invocations, lives for 1 hour
app = App(
    agent=agent,
    name="my_app",
    context_cache_config=ContextCacheConfig(
        cache_intervals=5,
        ttl_seconds=3600,
        min_tokens=8192,  # only cache when prior request > 8192 tokens
    ),
)
```

### Example 2 — with cache-creation timeout

```python
from google.genai import types

# Add a 10-second timeout on CachedContent.create() calls so a slow
# Google API doesn't block the entire request — on timeout the agent
# falls back to processing without a cache.
app = App(
    agent=agent,
    name="my_app",
    context_cache_config=ContextCacheConfig(
        ttl_seconds=1800,
        create_http_options=types.HttpOptions(timeout=10_000),  # 10s in ms
    ),
)
```

### Example 3 — disable caching for short sessions

```python
# No ContextCacheConfig → caching disabled entirely
app_no_cache = App(agent=agent, name="my_app")

# Or: set a high min_tokens so most sessions skip the cache
app_selective = App(
    agent=agent,
    name="my_app",
    context_cache_config=ContextCacheConfig(
        min_tokens=16_384,  # only cache when prior request > 16k tokens
    ),
)
```

---

## 6 · `LlmEventSummarizer`

**Source:** `google.adk.apps.llm_event_summarizer`

`LlmEventSummarizer` is a concrete implementation of `BaseEventsSummarizer`. It calls an LLM to compress a window of session events into a single `EventCompaction` — the core of ADK's sliding-window context compaction strategy.

The compactor only **generates** summaries; deciding *when* to trigger compaction and *which* events to include is handled by the `Runner` based on `EventsCompactionConfig`.

### Constructor (source-verified)

```python
class LlmEventSummarizer(BaseEventsSummarizer):
    _DEFAULT_PROMPT_TEMPLATE = (
        "The following is a conversation history between a user and an AI agent."
        " It may or may not start from a compacted history. Please identify and"
        " reiterate the user request, summarize the context so far, focusing on"
        " key decisions made and information obtained, as well as any unresolved"
        " questions or tasks. The summary should be concise and capture the"
        " essence of the interaction.\n\n{conversation_history}"
    )
    _MAX_TOOL_CONTENT_CHARS = 2000  # tool call args / responses are capped here

    def __init__(
        self,
        llm: BaseLlm,
        prompt_template: Optional[str] = None,
    ): ...
```

### Event formatting logic (source-verified)

`_format_events_for_prompt` converts the event list to a readable string:

- `part.thought and part.text` → `"{author} (thought): {text}"` — **skipped if the event already IS a compaction** (prevents summaries of summaries from leaking).
- `part.text` → `"{author}: {text}"`
- `part.function_call` → `"{author} called tool: {name}({args})"` — args truncated to 2000 chars.
- `part.function_response` → `"Tool response from {name}: {response}"` — response truncated to 2000 chars.

The 2000-character cap on tool content prevents large search results or API responses from inflating the compaction prompt beyond what the summary LLM can handle efficiently.

### `maybe_summarize_events` (source-verified)

```python
async def maybe_summarize_events(self, *, events: list[Event]) -> Optional[Event]:
    if not events:
        return None

    # Format the conversation window into a single string
    conversation_history = self._format_events_for_prompt(events)
    prompt = self._prompt_template.format(conversation_history=conversation_history)

    # Call the summarisation LLM (non-streaming)
    llm_request = LlmRequest(
        model=self._llm.model,
        contents=[Content(role="user", parts=[Part(text=prompt)])],
    )
    summary_content = None
    async for llm_response in self._llm.generate_content_async(llm_request, stream=False):
        if llm_response.content:
            summary_content = llm_response.content
            break

    if summary_content is None:
        return None

    # Wrap as EventCompaction spanning the provided window
    summary_content.role = "model"
    compaction = EventCompaction(
        start_timestamp=events[0].timestamp,
        end_timestamp=events[-1].timestamp,
        compacted_content=summary_content,
    )
    return Event(
        author="user",
        actions=EventActions(compaction=compaction),
        invocation_id=Event.new_id(),
        usage_metadata=summary_usage_metadata,
    )
```

### Example 1 — attach to an App with sliding-window compaction

```python
from google.adk.apps import App
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.agents import LlmAgent

agent = LlmAgent(name="assistant", model="gemini-2.0-flash")

# Create a summariser using a cheaper/faster model
summariser_llm = Gemini(model="gemini-2.0-flash-lite")
summariser = LlmEventSummarizer(llm=summariser_llm)

app = App(
    agent=agent,
    name="long_conv_app",
    events_compaction_config=EventsCompactionConfig(
        compaction_invocation_threshold=20,  # compact after 20 invocations
        overlap_size=3,                      # keep 3 most recent invocations after compaction
    ),
    events_summarizer=summariser,
)
```

### Example 2 — custom prompt template

```python
# Override the default prompt to produce a more structured summary
custom_template = (
    "Conversation history:\n{conversation_history}\n\n"
    "Produce a JSON summary with keys: "
    "'user_goal', 'completed_steps', 'pending_steps', 'key_facts'."
)

custom_summariser = LlmEventSummarizer(
    llm=Gemini(model="gemini-2.0-flash"),
    prompt_template=custom_template,
)
```

### Example 3 — standalone compaction (unit test)

```python
import asyncio
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.models.google_llm import Gemini
from google.genai import types

async def test_compaction():
    summariser = LlmEventSummarizer(llm=Gemini(model="gemini-2.0-flash"))

    events = [
        Event(
            author="user",
            invocation_id="inv1",
            content=types.Content(role="user", parts=[types.Part.from_text("Search for ADK docs")]),
        ),
        Event(
            author="assistant",
            invocation_id="inv1",
            content=types.Content(role="model", parts=[types.Part.from_text("Found 5 results...")]),
        ),
    ]

    compacted_event = await summariser.maybe_summarize_events(events=events)
    if compacted_event:
        print("Summary:", compacted_event.actions.compaction.compacted_content)

asyncio.run(test_compaction())
```

---

## 7 · `ToolConfirmation`

**Source:** `google.adk.tools.tool_confirmation`

`ToolConfirmation` is a small Pydantic model representing a request for human confirmation before a tool completes its action. When a tool function returns a `ToolConfirmation` instead of a normal result, the framework pauses execution and waits for the user to confirm or reject the action.

It is decorated with `@experimental(FeatureName.TOOL_CONFIRMATION)`.

### Field reference (source-verified)

```python
@experimental(FeatureName.TOOL_CONFIRMATION)
class ToolConfirmation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        alias_generator=alias_generators.to_camel,
        populate_by_name=True,
    )

    hint: str = ""
    """Human-readable explanation of why confirmation is needed."""

    confirmed: bool = False
    """Whether execution is confirmed. Set to True by the HITL handler."""

    payload: Optional[Any] = None
    """Optional JSON-serializable custom data. Useful for passing UI context."""
```

The camelCase alias generator means the model serialises as `{"hint": "...", "confirmed": false, "payload": null}` or `{"hint": "...", "confirmed": false, "payload": null}` — and can be populated from camelCase JSON inputs as well.

### How it works

1. A tool function returns `ToolConfirmation(hint="...", confirmed=False, payload={...})`.
2. The framework detects the `ToolConfirmation` return type and emits a `RequestInput` interrupt.
3. The human reviews the `hint` and `payload`, then resumes with `confirmed=True`.
4. The framework calls the tool again; the second call proceeds normally.

### Example 1 — simple approval gate

```python
from google.adk.tools.tool_confirmation import ToolConfirmation

async def delete_user_account(user_id: str) -> dict | ToolConfirmation:
    """Delete a user account. Requires explicit confirmation."""
    # Always return ToolConfirmation on the first call
    return ToolConfirmation(
        hint=f"Are you sure you want to permanently delete user '{user_id}'?",
        confirmed=False,
        payload={"user_id": user_id, "action": "delete"},
    )
```

### Example 2 — conditional confirmation (confirm only for high-risk actions)

```python
SENSITIVE_TABLES = {"payments", "user_credentials", "audit_log"}

async def run_sql(query: str, tool_context: ToolContext) -> dict | ToolConfirmation:
    """Execute a SQL query. Requires confirmation for writes to sensitive tables."""
    is_write = any(kw in query.upper() for kw in ("DELETE", "DROP", "TRUNCATE", "UPDATE"))
    touches_sensitive = any(t in query.lower() for t in SENSITIVE_TABLES)

    if is_write and touches_sensitive:
        return ToolConfirmation(
            hint=f"This query modifies a sensitive table. Query: {query[:200]}",
            confirmed=False,
            payload={"query": query},
        )

    # Safe query — execute directly
    return await execute_query_impl(query)
```

### Example 3 — using `ToolConfirmation` with `FunctionTool`

```python
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.agents import LlmAgent

async def send_email(to: str, subject: str, body: str) -> dict | ToolConfirmation:
    """Send an email. Always requires confirmation before sending."""
    return ToolConfirmation(
        hint=f"Send email to '{to}' with subject '{subject}'?",
        confirmed=False,
        payload={"to": to, "subject": subject, "body_preview": body[:100]},
    )

agent = LlmAgent(
    name="email_agent",
    model="gemini-2.0-flash",
    tools=[send_email],
    instruction=(
        "Draft and send emails on behalf of the user. "
        "Always confirm before actually sending."
    ),
)
```

---

## 8 · `ToolboxToolset`

**Source:** `google.adk.tools.toolbox_toolset`

`ToolboxToolset` connects an ADK agent to a running [MCP Toolbox for Databases](https://github.com/googleapis/mcp-toolbox-sdk-python) server. It delegates all tool loading to `toolbox_adk.ToolboxToolset` (a separate package), exposing its full API through ADK's `BaseToolset` interface.

### Installation

```bash
pip install "google-adk[toolbox]"
# installs the optional toolbox-adk dependency
```

### Constructor (source-verified)

```python
class ToolboxToolset(BaseToolset):
    def __init__(
        self,
        server_url: str,
        toolset_name: Optional[str] = None,
        tool_names: Optional[List[str]] = None,
        auth_token_getters: Optional[Mapping[str, Callable[[], str]]] = None,
        bound_params: Optional[Mapping[str, Union[Callable[[], Any], Any]]] = None,
        credentials: Optional[CredentialConfig] = None,
        additional_headers: Optional[Mapping[str, str]] = None,
        **kwargs,
    ): ...
```

| Parameter | Purpose |
|---|---|
| `server_url` | URL of the Toolbox server (e.g. `"http://127.0.0.1:5000"`) |
| `toolset_name` | Load a named toolset from the server; `None` → all tools |
| `tool_names` | Load specific named tools; combined with `toolset_name` (union, not intersection) |
| `auth_token_getters` | Dict of auth service name → `Callable[[], str]`; provides tokens for server-side auth |
| `bound_params` | Dict of param name → value or `Callable[[], value]`; binds params so they are not exposed to the LLM |
| `credentials` | `toolbox_adk.CredentialConfig` for secure server authentication |
| `additional_headers` | Static headers added to every Toolbox API call |

The constructor raises `ImportError` with a clear message if `toolbox-adk` is not installed.

### Example 1 — minimal setup

```python
from google.adk.tools.toolbox_toolset import ToolboxToolset
from google.adk.agents import LlmAgent

toolbox = ToolboxToolset(server_url="http://localhost:5000")

agent = LlmAgent(
    name="db_agent",
    model="gemini-2.0-flash",
    toolsets=[toolbox],
    instruction="You have access to database tools. Answer questions by querying the database.",
)
```

### Example 2 — named toolset with bound parameters

```python
import os

toolbox = ToolboxToolset(
    server_url="http://localhost:5000",
    toolset_name="customer-tools",
    # Bind the user_id so it's never exposed to or settable by the LLM
    bound_params={
        "user_id": lambda: get_current_user_id(),      # dynamic callable
        "region": "us-east1",                           # static value
    },
)
```

### Example 3 — with auth token getter for secured server

```python
import google.auth
import google.auth.transport.requests

def get_google_id_token() -> str:
    """Get an OIDC token for authenticating to the Toolbox server."""
    credentials, _ = google.auth.default()
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token

toolbox = ToolboxToolset(
    server_url="https://toolbox.my-project.run.app",
    auth_token_getters={"google": get_google_id_token},
)
```

### Example 4 — selective tool loading

```python
toolbox = ToolboxToolset(
    server_url="http://localhost:5000",
    tool_names=["search_products", "get_product_details"],
    # Only these two tools are loaded, even if the server has more
)
```

---

## 9 · `DynamicNodeScheduler` + `DynamicNodeRun` + `DynamicNodeState`

**Source:** `google.adk.workflow._dynamic_node_scheduler`

`DynamicNodeScheduler` implements the `ctx.run_node()` call for `Workflow`. When a workflow node calls `ctx.run_node(some_node, ...)`, the scheduler handles three cases:

1. **Dedup** — the node already ran this invocation (task is running or completed): return the cached result.
2. **Resume** — the node ran in a *previous* invocation (found in session events): fast-forward or re-run based on interrupt/completion state.
3. **Fresh** — the node has never run: execute it normally.

### `DynamicNodeRun` (source-verified)

```python
@dataclass(kw_only=True)
class DynamicNodeRun:
    state: NodeState       # tracking status, run_id, interrupts
    output: Any = None     # final output once the node completes
    task: asyncio.Task[Context] | None = None
    transfer_to_agent: str | None = None
    recovered_state: _ChildScanState | None = None  # raw scan state from events
```

### `DynamicNodeState` (source-verified)

```python
@dataclass(kw_only=True)
class DynamicNodeState:
    runs: dict[str, DynamicNodeRun] = field(default_factory=dict)
    # key = full node_path string, e.g. "/wf@1/node_a@1"

    interrupt_ids: set[str] = field(default_factory=set)
    # union of all unresolved interrupt IDs from static + dynamic child nodes

    def get_dynamic_tasks(self) -> list[asyncio.Task[Context]]: ...
```

### Node path convention

Every dynamic node run gets a path derived from the parent node's path plus the node name and run ID:

```
/workflow@1/parent_node@1/child_node@{run_id}
```

`run_id` is auto-assigned from a per-parent counter: the first call to `ctx.run_node(target)` uses `run_id="1"`, the second `"2"`, and so on. Callers can also pass a custom `run_id` for deterministic replay.

### The scheduling algorithm (source-verified)

```
__call__():
  1. Build node_path from parent_ctx.node_path + name@run_id
  2. Validate input against node.input_schema
  3. [Phase 1] If node_path not in state.runs: scan session events (lazy rehydration)
  4. [Phase 2] Check existing run:
       - Task running → await existing task (dedup)
       - Recovered + all interrupts resolved → fast-forward (no re-run)
       - Recovered + unresolved interrupts → bubble interrupts up
       - Should re-run → re-run with resume_inputs
  5. [Phase 3] If no existing run: fresh execution via asyncio.create_task()
  6. If result has transfer_to_agent → resolve target agent and loop
```

### Example 1 — calling `ctx.run_node()` inside a workflow node

```python
from google.adk.workflow import Workflow
from google.adk.agents import LlmAgent

researcher = LlmAgent(name="researcher", model="gemini-2.0-flash",
                      instruction="Research the given topic and return a brief summary.")
writer = LlmAgent(name="writer", model="gemini-2.0-flash",
                  instruction="Write a blog post based on the provided research.")

@workflow.node
async def orchestrate(ctx):
    # Run researcher and writer sequentially as dynamic nodes
    research_ctx = await ctx.run_node(researcher, node_input="quantum computing")
    summary = research_ctx.output

    # Pass summary to writer
    article_ctx = await ctx.run_node(writer, node_input=summary)
    return article_ctx.output

workflow = Workflow(nodes=[orchestrate], name="article_workflow")
```

### Example 2 — parallel dynamic node fan-out

```python
import asyncio

@workflow.node
async def parallel_research(ctx):
    topics = ["LLMs", "Agents", "RAG", "MCP"]
    # Schedule all researcher nodes concurrently
    tasks = [
        ctx.run_node(researcher, node_input=topic, node_name=f"research_{topic}")
        for topic in topics
    ]
    results = await asyncio.gather(*tasks)
    return [r.output for r in results]
```

### Example 3 — transfer_to_agent handling

```python
# DynamicNodeScheduler resolves transfer_to_agent automatically.
# If node output contains actions.transfer_to_agent = "some_agent",
# the scheduler resolves the target agent from the workflow's sub-agents,
# re-runs with the target, and loops until a non-transfer result is returned.
# No extra code needed in the node — the scheduler handles it transparently.
```

### Example 4 — resuming across turns (HITL)

```python
@workflow.node
async def approval_step(ctx):
    # Raises NodeInterruptedError when a RequestInput is returned by a tool
    # On resume (next ADK invocation with same session_id), DynamicNodeScheduler
    # rehydrates the node's state from session events, detects that the interrupt
    # was resolved, and re-runs the node with resume_inputs populated.
    result = await ctx.run_node(
        approval_agent,
        node_input={"document": ctx.node_input},
        rerun_on_resume=True,  # re-execute the node after HITL approval
    )
    return result.output
```

---

## 10 · `FileArtifactService`

**Source:** `google.adk.artifacts.file_artifact_service`

`FileArtifactService` is a `BaseArtifactService` implementation that stores artifacts as files on the local filesystem. It provides the same versioned, session/user-scoped artifact API as `GcsArtifactService` and `InMemoryArtifactService`, making it ideal for local development and testing.

### Constructor (source-verified)

```python
class FileArtifactService(BaseArtifactService):
    def __init__(self, root_dir: Path | str):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
```

### Storage layout (source-verified from module docstring)

```
root_dir/
└── users/
    └── {user_id}/
        ├── sessions/
        │   └── {session_id}/
        │       └── artifacts/
        │           └── {artifact_path}/     # derived from filename
        │               └── versions/
        │                   └── {version}/
        │                       ├── {stored_filename}
        │                       └── metadata.json
        └── artifacts/                       # user-scoped (no session)
            └── {artifact_path}/...
```

### Scoping: session vs user

Artifacts are either **session-scoped** (default) or **user-scoped** (persisted across sessions):

| Trigger | Storage location |
|---|---|
| `filename` starts with `"user:"` | User scope: `root/users/{user_id}/artifacts/` |
| `session_id` is `None` | User scope |
| Normal `filename` with `session_id` | Session scope: `root/users/{user_id}/sessions/{session_id}/artifacts/` |

```python
# Session-scoped artifact — only visible in this session
await service.save_artifact(
    app_name="app", user_id="u1", session_id="s1",
    filename="report.pdf",
    artifact=types.Part(inline_data=types.Blob(mime_type="application/pdf", data=b"...")),
)

# User-scoped artifact — persists across all sessions for user "u1"
await service.save_artifact(
    app_name="app", user_id="u1", session_id="s1",
    filename="user:shared/config.json",      # "user:" prefix forces user scope
    artifact=types.Part(text='{"theme": "dark"}'),
)
```

### Path traversal protection (source-verified)

`_resolve_scoped_artifact_path` rejects filenames that escape the storage scope:

```python
# These all raise InputValidationError:
save_artifact(filename="../../secret.txt")   # traversal
save_artifact(filename="/etc/passwd")        # absolute path
save_artifact(filename="")                   # empty after stripping
```

Separators in the filename **create nested directories**, so `"images/photo.png"` stores at `.../artifacts/images/photo/versions/0/photo`.

### Versioning (source-verified)

Each `save_artifact` call creates a new version directory (`0`, `1`, `2`, …). Loading without specifying a version returns the latest:

```python
v0 = await service.save_artifact(...)  # returns 0
v1 = await service.save_artifact(...)  # returns 1

part = await service.load_artifact(..., version=None)   # loads v1 (latest)
part = await service.load_artifact(..., version=0)      # loads v0 explicitly

versions = await service.list_versions(...)  # [0, 1]
```

### `metadata.json` sidecar (source-verified)

Each version directory contains a `metadata.json` sidecar with the `FileArtifactVersion` model:

```python
class FileArtifactVersion(ArtifactVersion):
    file_name: str          # original filename from the caller
    display_name: Optional[str]  # from inline_data.display_name
    # Inherits: version, mime_type, canonical_uri (file:// URI), custom_metadata
```

### Example 1 — local dev setup

```python
from pathlib import Path
from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.adk.apps import App
from google.adk.agents import LlmAgent

artifact_service = FileArtifactService(root_dir=Path("~/.adk/artifacts"))

app = App(
    agent=LlmAgent(name="assistant", model="gemini-2.0-flash"),
    name="my_app",
    artifact_service=artifact_service,
)
```

### Example 2 — saving and loading a text report

```python
import asyncio
from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.genai import types

async def save_and_load():
    service = FileArtifactService(root_dir="/tmp/adk_artifacts")

    # Save a text artifact
    version = await service.save_artifact(
        app_name="demo",
        user_id="alice",
        session_id="sess123",
        filename="quarterly_report.txt",
        artifact=types.Part(text="Q1 2026: Revenue up 15%..."),
    )
    print(f"Saved as version {version}")

    # Load the latest version
    part = await service.load_artifact(
        app_name="demo",
        user_id="alice",
        session_id="sess123",
        filename="quarterly_report.txt",
    )
    print(part.text)

asyncio.run(save_and_load())
```

### Example 3 — listing all artifacts for a session

```python
async def list_session_artifacts(service, user_id, session_id):
    keys = await service.list_artifact_keys(
        app_name="demo",
        user_id=user_id,
        session_id=session_id,
    )
    # Returns sorted list of filename strings, e.g.:
    # ["quarterly_report.txt", "user:shared/config.json"]
    for key in keys:
        versions = await service.list_versions(
            app_name="demo", user_id=user_id,
            session_id=session_id, filename=key,
        )
        print(f"{key}: {len(versions)} version(s)")
```

### Example 4 — binary artifact (image)

```python
async def save_image(service, image_bytes: bytes):
    version = await service.save_artifact(
        app_name="demo",
        user_id="alice",
        session_id="sess123",
        filename="screenshot.png",
        artifact=types.Part(
            inline_data=types.Blob(
                mime_type="image/png",
                data=image_bytes,
                display_name="screenshot.png",
            )
        ),
    )
    # Stored at: root/users/alice/sessions/sess123/artifacts/screenshot.png/
    #            versions/0/screenshot.png  (binary)
    #            versions/0/metadata.json   (sidecar)
    return version
```

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-06-24 | 2.3.0 | Vol. 26 added: `LangGraphAgent`, `PubSubToolset`, `SpannerToolset`/`SpannerVectorStoreSettings`, `LongRunningFunctionTool`, `ContextCacheConfig`, `LlmEventSummarizer`, `ToolConfirmation`, `ToolboxToolset`, `DynamicNodeScheduler`, `FileArtifactService` — all source-verified against installed `google-adk==2.3.0`. |

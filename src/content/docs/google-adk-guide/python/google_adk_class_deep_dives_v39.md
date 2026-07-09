---
title: "Class deep dives — volume 39 (10 new APIs)"
description: "Source-verified deep dives into 10 google-adk 2.4.0 APIs: SpannerToolset+SpannerAdminToolset+SpannerCredentialsConfig (experimental Spanner integration; 8 data tools + 7 admin tools; GoogleTool wrapper; FeatureName guards), SpannerToolSettings+SpannerVectorStoreSettings+VectorSearchIndexSettings+TableColumn (SQL query modes; ANN vs exact vector search; tree_depth/num_leaves/num_branches; primary_key_columns), UserSimulator+NextUserMessage+Status+BaseUserSimulatorConfig (abstract evaluation simulator ABC; SUCCESS/TURN_LIMIT_REACHED/STOP_SIGNAL_DETECTED/NO_MESSAGE_GENERATED; get_simulation_evaluator hook), StaticUserSimulator (StaticConversation replay; invocation_idx counter; STOP_SIGNAL_DETECTED on exhaustion), LlmBackedUserSimulator+LlmBackedUserSimulatorConfig (LLM-driven simulation; max_allowed_invocations=20 runaway guard; </finished> stop signal; thinking_budget=10240; custom_instructions Jinja template), CachePerformanceAnalyzer (async event-history analytics; cache_hit_ratio_percent; cache_utilization_ratio_percent; unique cache_refreshes set count), ContextCacheConfig (app-level caching; cache_intervals=10; ttl_seconds=1800; Gemini 4096-token hard minimum; 2nd-turn gate; create_http_options timeout; ttl_string property), TelemetryContext+start_as_current_node_span (OTel span lifecycle; invoke_agent/invoke_workflow/invoke_node naming; GEN_AI_WORKFLOW_NESTED nested flag; _ENTRYPOINT_WORKFLOW_KEY ContextVar detection; associated_event_ids stamping), JoinNode+Trigger (fan-in barrier; _requires_all_predecessors=True; aggregated dict validation; Trigger input/use_sub_branch/branch/isolation_scope; ser_json_bytes=base64), Graph+Edge+RouteValue (workflow graph construction; from_edge_items; model_post_init node inference; get_next_pending_nodes route matching; DEFAULT_ROUTE fallback; RoutingMap fan-out)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 39"
  order: 108
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source (locate yours with
`python -c 'import google.adk; print(google.adk.__file__)'`) on
**google-adk == 2.4.0**. No documentation or blog posts were used as primary
sources.
</Aside>

---

## 1 · `SpannerToolset` + `SpannerAdminToolset` + `SpannerCredentialsConfig` — Spanner integration toolsets

**Source:** `google/adk/tools/spanner/spanner_toolset.py`, `admin_toolset.py`, `spanner_credentials.py`

Both toolsets are gated behind `@experimental` feature flags (`FeatureName.SPANNER_TOOLSET` and `FeatureName.SPANNER_ADMIN_TOOLSET`). They wrap Spanner operations as `GoogleTool` instances, passing shared `SpannerCredentialsConfig` and `SpannerToolSettings` through every tool.

### Key implementation details

```python
SPANNER_DEFAULT_SCOPE = [
    "https://www.googleapis.com/auth/spanner.admin",
    "https://www.googleapis.com/auth/spanner.data",
]

DEFAULT_SPANNER_TOOL_NAME_PREFIX = "spanner"

@experimental(FeatureName.SPANNER_TOOLSET)
class SpannerToolset(BaseToolset):
    # Tool names: spanner_list_table_names, spanner_list_table_indexes,
    #   spanner_list_table_index_columns, spanner_list_named_schemas,
    #   spanner_get_table_schema, spanner_execute_sql,
    #   spanner_similarity_search, spanner_vector_store_similarity_search

    def __init__(
        self,
        *,
        tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
        credentials_config: Optional[SpannerCredentialsConfig] = None,
        spanner_tool_settings: Optional[SpannerToolSettings] = None,
    ): ...

@experimental(FeatureName.SPANNER_ADMIN_TOOLSET)
class SpannerAdminToolset(BaseToolset):
    # Tool names: spanner_list_instances, spanner_get_instance,
    #   spanner_create_database, spanner_list_databases,
    #   spanner_create_instance, spanner_list_instance_configs,
    #   spanner_get_instance_config
    ...

@experimental(FeatureName.GOOGLE_CREDENTIALS_CONFIG)
class SpannerCredentialsConfig(BaseGoogleCredentialsConfig):
    SPANNER_TOKEN_CACHE_KEY = "spanner_token_cache"
    # __post_init__ sets default scopes to SPANNER_DEFAULT_SCOPE when None
    # and sets _token_cache_key = SPANNER_TOKEN_CACHE_KEY
```

`SpannerToolset.get_tools()` always includes the 5 metadata tools and conditionally adds `spanner_execute_sql` and `spanner_similarity_search` when `Capabilities.DATA_READ` is in `tool_settings.capabilities`. The `spanner_vector_store_similarity_search` tool is only added when `vector_store_settings` is non-`None`. Tools are then filtered through `_is_tool_selected()`, which supports both `ToolPredicate` callables and allowlist strings.

### Example 1 — read-only Spanner data toolset

```python
from google.adk.tools.spanner.spanner_toolset import SpannerToolset
from google.adk.tools.spanner.settings import SpannerToolSettings

settings = SpannerToolSettings(
    max_executed_query_result_rows=20,
)

# Tool filter by name allowlist
toolset = SpannerToolset(
    tool_filter=["spanner_execute_sql", "spanner_get_table_schema"],
    spanner_tool_settings=settings,
)

import asyncio

async def main():
    tools = await toolset.get_tools()
    print([t.name for t in tools])
    # ['spanner_get_table_schema', 'spanner_execute_sql']

asyncio.run(main())
```

### Example 2 — service-account credentials with custom scopes

```python
from google.oauth2 import service_account
from google.adk.tools.spanner.spanner_credentials import (
    SpannerCredentialsConfig,
    SPANNER_DEFAULT_SCOPE,
)
from google.adk.tools.spanner.spanner_toolset import SpannerToolset

# Load the SA key file into a Credentials object first — SpannerCredentialsConfig
# accepts a loaded credentials object via the `credentials` field (extra="forbid"
# means no other kwargs are accepted alongside it).
sa_creds = service_account.Credentials.from_service_account_file(
    "/path/to/sa.json",
    scopes=SPANNER_DEFAULT_SCOPE,
)

creds_config = SpannerCredentialsConfig(credentials=sa_creds)
# SpannerCredentialsConfig.__post_init__ also sets SPANNER_DEFAULT_SCOPE when
# scopes is None, but passing scopes= to from_service_account_file is clearer.

toolset = SpannerToolset(credentials_config=creds_config)
```

### Example 3 — admin toolset for instance management

```python
from google.adk.tools.spanner.admin_toolset import SpannerAdminToolset
from google.adk.tools.spanner.settings import SpannerToolSettings

admin = SpannerAdminToolset(
    spanner_tool_settings=SpannerToolSettings(),
)

async def list_admin_tools():
    tools = await admin.get_tools()
    return [t.name for t in tools]
    # ['spanner_list_instances', 'spanner_get_instance',
    #  'spanner_create_database', 'spanner_list_databases',
    #  'spanner_create_instance', 'spanner_list_instance_configs',
    #  'spanner_get_instance_config']
```

---

## 2 · `SpannerToolSettings` + `SpannerVectorStoreSettings` + `VectorSearchIndexSettings` + `TableColumn` — Spanner tool configuration

**Source:** `google/adk/tools/spanner/settings.py`

`SpannerToolSettings` is the top-level configuration object injected into all Spanner tools. It composes with `SpannerVectorStoreSettings` for vector similarity search and with `VectorSearchIndexSettings` for ANN index control.

### Key implementation details

```python
class Capabilities(Enum):
    DATA_READ = "data_read"   # Read-only data operations (default)

class QueryResultMode(Enum):
    DEFAULT = "default"       # Returns list of rows
    DICT_LIST = "dict_list"   # Returns list of {column: value} dicts

class TableColumn(BaseModel):
    name: str
    type: str           # e.g. 'STRING(MAX)', 'INT64', 'text', 'int8'
    is_nullable: bool = True

class VectorSearchIndexSettings(BaseModel):
    index_name: str
    additional_key_columns: Optional[list[str]] = None   # selective pre-filter
    additional_storing_columns: Optional[list[str]] = None
    tree_depth: int = 2                   # 2 or 3; use 3 for >100M rows
    num_leaves: int = 1000                # recommended: num_rows / 1000
    num_branches: Optional[int] = None   # only for tree_depth=3

EXACT_NEAREST_NEIGHBORS = "EXACT_NEAREST_NEIGHBORS"
APPROXIMATE_NEAREST_NEIGHBORS = "APPROXIMATE_NEAREST_NEIGHBORS"

@experimental(FeatureName.SPANNER_TOOL_SETTINGS)
class SpannerToolSettings(BaseModel):
    capabilities: List[Capabilities] = [Capabilities.DATA_READ]
    max_executed_query_result_rows: int = 50
    query_result_mode: QueryResultMode = QueryResultMode.DEFAULT
    database_role: Optional[str] = None
    vector_store_settings: Optional[SpannerVectorStoreSettings] = None

class SpannerVectorStoreSettings(BaseModel):
    project_id: str
    instance_id: str
    database_id: str
    table_name: str
    content_column: str
    embedding_column: str
    vector_length: int            # must match model output dim
    vertex_ai_embedding_model_name: str   # e.g. 'text-embedding-005'
    selected_columns: list[str] = []     # defaults to [content_column]
    nearest_neighbors_algorithm: NearestNeighborsAlgorithm = "EXACT_NEAREST_NEIGHBORS"
    top_k: int = 4
    distance_type: str = "COSINE"        # COSINE, DOT_PRODUCT, EUCLIDEAN
    num_leaves_to_search: Optional[int] = None  # ANN only
    additional_filter: Optional[str] = None
    vector_search_index_settings: Optional[VectorSearchIndexSettings] = None
    additional_columns_to_setup: Optional[list[TableColumn]] = None
    primary_key_columns: Optional[list[str]] = None  # defaults to auto-UUID 'id'
```

The `SpannerVectorStoreSettings` model uses a `@model_validator(mode="after")` to enforce `vector_length > 0`, auto-populate `selected_columns` with `content_column` if empty, and verify that every declared `primary_key_column` appears in the column definitions (`content_column`, `embedding_column`, and any `additional_columns_to_setup`). An invalid entry raises `ValueError`.

### Example 1 — dict-list query mode with row cap

```python
from google.adk.tools.spanner.settings import SpannerToolSettings, QueryResultMode

settings = SpannerToolSettings(
    max_executed_query_result_rows=10,
    query_result_mode=QueryResultMode.DICT_LIST,
)
# Each row is {"col": value, ...} instead of a plain list
```

### Example 2 — exact nearest-neighbor vector search

```python
from google.adk.tools.spanner.settings import (
    SpannerToolSettings, SpannerVectorStoreSettings
)

vss = SpannerVectorStoreSettings(
    project_id="my-project",
    instance_id="my-instance",
    database_id="my-db",
    table_name="documents",
    content_column="text",
    embedding_column="embedding",
    vector_length=768,
    vertex_ai_embedding_model_name="text-embedding-005",
    top_k=5,
    distance_type="DOT_PRODUCT",
)

settings = SpannerToolSettings(vector_store_settings=vss)
```

### Example 3 — ANN index with 3-level tree and custom columns

```python
from google.adk.tools.spanner.settings import (
    SpannerVectorStoreSettings, VectorSearchIndexSettings, TableColumn
)

idx = VectorSearchIndexSettings(
    index_name="documents_embedding_idx",
    tree_depth=3,        # for >100M rows
    num_leaves=2000,
    num_branches=50,
    additional_storing_columns=["category"],   # early filter
)

vss = SpannerVectorStoreSettings(
    project_id="my-project",
    instance_id="my-instance",
    database_id="my-db",
    table_name="documents",
    content_column="text",
    embedding_column="embedding",
    vector_length=768,
    vertex_ai_embedding_model_name="text-embedding-005",
    nearest_neighbors_algorithm="APPROXIMATE_NEAREST_NEIGHBORS",
    num_leaves_to_search=100,
    additional_filter="category = 'research'",
    vector_search_index_settings=idx,
    additional_columns_to_setup=[
        TableColumn(name="category", type="STRING(MAX)", is_nullable=False),
    ],
    primary_key_columns=["category"],   # must be in additional_columns_to_setup
)
```

---

## 3 · `UserSimulator` + `NextUserMessage` + `Status` + `BaseUserSimulatorConfig` — evaluation simulator ABC

**Source:** `google/adk/evaluation/simulation/user_simulator.py`

`UserSimulator` is the abstract base class for automated evaluation; implementations receive the conversation history and return the next simulated user turn. The `@experimental` decorator flags the API as unstable.

### Key implementation details

```python
class Status(enum.Enum):
    SUCCESS = "success"
    TURN_LIMIT_REACHED = "turn_limit_reached"
    STOP_SIGNAL_DETECTED = "stop_signal_detected"
    NO_MESSAGE_GENERATED = "no_message_generated"

class NextUserMessage(EvalBaseModel):
    status: Status
    user_message: Optional[genai_types.Content] = None

    @model_validator(mode="after")
    def ensure_user_message_iff_success(self) -> NextUserMessage:
        # Invariant: user_message is non-None iff status is SUCCESS
        if (self.status == Status.SUCCESS) == (self.user_message is None):
            raise ValueError("user_message provided iff status==SUCCESS")
        return self

class BaseUserSimulatorConfig(BaseModel):
    model_config = ConfigDict(
        alias_generator=alias_generators.to_camel,  # camelCase JSON support
        populate_by_name=True,
        extra="allow",
    )

@experimental
class UserSimulator(ABC):
    def __init__(self, config, config_type):
        # Unpacks config into the declared type via model_validate + model_dump
        self._config = config_type.model_validate(config.model_dump())

    async def get_next_user_message(self, events: list[Event]) -> NextUserMessage:
        raise NotImplementedError()

    def get_simulation_evaluator(self) -> Optional[Evaluator]:
        raise NotImplementedError()
```

Callers inspect `NextUserMessage.status` before using `user_message`. Only `SUCCESS` produces a content object; the other statuses signal the runner to stop the simulation loop.

### Example 1 — custom simulator returning a fixed greeting

```python
from google.adk.evaluation.simulation.user_simulator import (
    UserSimulator, BaseUserSimulatorConfig, NextUserMessage, Status
)
from google.genai import types

class GreetingSimulator(UserSimulator):
    def __init__(self):
        super().__init__(BaseUserSimulatorConfig(), BaseUserSimulatorConfig)
        self._sent = False

    async def get_next_user_message(self, events):
        if self._sent:
            return NextUserMessage(status=Status.STOP_SIGNAL_DETECTED)
        self._sent = True
        return NextUserMessage(
            status=Status.SUCCESS,
            user_message=types.Content(
                role="user",
                parts=[types.Part(text="Hello, agent!")],
            ),
        )

    def get_simulation_evaluator(self):
        return None
```

### Example 2 — checking the model_validator invariant

```python
from google.adk.evaluation.simulation.user_simulator import NextUserMessage, Status

# SUCCESS requires user_message
try:
    NextUserMessage(status=Status.SUCCESS, user_message=None)
except ValueError as e:
    print(e)  # user_message provided iff status==SUCCESS

# Non-SUCCESS must not have user_message
from google.genai import types
try:
    NextUserMessage(
        status=Status.TURN_LIMIT_REACHED,
        user_message=types.Content(role="user", parts=[types.Part(text="hi")]),
    )
except ValueError as e:
    print(e)  # same validator fires
```

### Example 3 — config type coercion in constructor

```python
from pydantic import BaseModel
from google.adk.evaluation.simulation.user_simulator import (
    UserSimulator, BaseUserSimulatorConfig, NextUserMessage, Status
)

class MyConfig(BaseUserSimulatorConfig):
    persona: str = "helpful assistant"

class PersonaSimulator(UserSimulator):
    def __init__(self, persona: str = "helpful assistant"):
        cfg = MyConfig(persona=persona)
        super().__init__(cfg, MyConfig)
        # self._config is MyConfig, coerced via model_validate(model_dump())

    async def get_next_user_message(self, events):
        return NextUserMessage(status=Status.NO_MESSAGE_GENERATED)

    def get_simulation_evaluator(self):
        return None

sim = PersonaSimulator(persona="skeptical user")
print(sim._config.persona)  # skeptical user
```

---

## 4 · `StaticUserSimulator` — replay-based evaluation simulator

**Source:** `google/adk/evaluation/simulation/static_user_simulator.py`

`StaticUserSimulator` plays back a pre-recorded `StaticConversation` (a sequence of invocations). It is the simplest concrete simulator: no LLM required, and the simulation evaluator is always `None`.

### Key implementation details

```python
@experimental
class StaticUserSimulator(UserSimulator):
    def __init__(self, *, static_conversation: StaticConversation):
        super().__init__(BaseUserSimulatorConfig(), BaseUserSimulatorConfig)
        self.static_conversation = static_conversation
        self.invocation_idx = 0

    async def get_next_user_message(self, events: list[Event]) -> NextUserMessage:
        if self.invocation_idx >= len(self.static_conversation):
            return NextUserMessage(status=Status.STOP_SIGNAL_DETECTED)
        next_user_content = self.static_conversation[self.invocation_idx].user_content
        self.invocation_idx += 1
        return NextUserMessage(status=Status.SUCCESS, user_message=next_user_content)

    def get_simulation_evaluator(self) -> Optional[Evaluator]:
        return None   # static replay needs no quality evaluator
```

`StaticConversation` is a `list[Invocation]`; `Invocation.user_content` is the `genai_types.Content` to send next. The `events` parameter is accepted but ignored — the static sequence is independent of what the agent actually said.

### Example 1 — two-turn static simulation

```python
import asyncio
from google.adk.evaluation.simulation.static_user_simulator import StaticUserSimulator
from google.adk.evaluation.eval_case import Invocation
from google.genai import types

def make_invocation(text: str) -> Invocation:
    return Invocation(
        user_content=types.Content(
            role="user",
            parts=[types.Part(text=text)],
        )
    )

sim = StaticUserSimulator(
    static_conversation=[
        make_invocation("What is the capital of France?"),
        make_invocation("And what is its population?"),
    ]
)

async def run():
    msg1 = await sim.get_next_user_message([])
    print(msg1.user_message.parts[0].text)   # What is the capital of France?
    msg2 = await sim.get_next_user_message([])
    print(msg2.user_message.parts[0].text)   # And what is its population?
    msg3 = await sim.get_next_user_message([])
    print(msg3.status)                        # Status.STOP_SIGNAL_DETECTED

asyncio.run(run())
```

### Example 2 — index counter advances independently of events

```python
import asyncio
from google.adk.evaluation.simulation.static_user_simulator import StaticUserSimulator
from google.adk.evaluation.eval_case import Invocation
from google.genai import types

sim = StaticUserSimulator(
    static_conversation=[
        Invocation(user_content=types.Content(role="user", parts=[types.Part(text="step 1")])),
    ]
)

async def demo():
    # Events from the agent do NOT affect the index — passed but ignored
    fake_events = []   # or any list of Event objects
    r = await sim.get_next_user_message(fake_events)
    print(r.status)   # Status.SUCCESS
    r2 = await sim.get_next_user_message(fake_events)
    print(r2.status)  # Status.STOP_SIGNAL_DETECTED

asyncio.run(demo())
```

### Example 3 — using evaluator (always None for static)

```python
from google.adk.evaluation.simulation.static_user_simulator import StaticUserSimulator
from google.adk.evaluation.eval_case import Invocation
from google.genai import types

sim = StaticUserSimulator(
    static_conversation=[
        Invocation(user_content=types.Content(role="user", parts=[types.Part(text="hi")])),
    ]
)
evaluator = sim.get_simulation_evaluator()
print(evaluator)   # None — no quality scoring for static replay
```

---

## 5 · `LlmBackedUserSimulator` + `LlmBackedUserSimulatorConfig` — LLM-driven evaluation simulator

**Source:** `google/adk/evaluation/simulation/llm_backed_user_simulator.py`

`LlmBackedUserSimulator` drives a real LLM (defaulting to `gemini-2.5-flash` with thinking enabled) to generate user turns dynamically, guided by a `ConversationScenario`. A `</finished>` stop signal in the model response ends the simulation.

### Key implementation details

```python
_STOP_SIGNAL = "</finished>"

class LlmBackedUserSimulatorConfig(BaseUserSimulatorConfig):
    model: str = "gemini-2.5-flash"
    model_configuration: genai_types.GenerateContentConfig = Field(
        default_factory=lambda: genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=10240,
            )
        )
    )
    max_allowed_invocations: int = 20   # -1 = unlimited (not recommended)
    custom_instructions: str | None = None  # Jinja template; field_validator requires
    # {{ stop_signal }}, {{ conversation_plan }}, AND {{ conversation_history }}

@experimental
class LlmBackedUserSimulator(UserSimulator):
    _AUTHOR_USER = "user"

    async def get_next_user_message(self, events: list[Event]) -> NextUserMessage:
        # Checks max_allowed_invocations → returns TURN_LIMIT_REACHED
        # Builds LlmRequest with system prompt + prior turns
        # Streams response; if STOP_SIGNAL in text → STOP_SIGNAL_DETECTED
        # Otherwise returns SUCCESS with the generated content
```

The `ConversationScenario` (injected at construction) provides the goal that shapes the system prompt. The per-invocation count starts at 1 (for the initial system prompt) to make the guard meaningful.

### Example 1 — minimal LLM-backed simulator

```python
from google.adk.evaluation.simulation.llm_backed_user_simulator import (
    LlmBackedUserSimulator,
    LlmBackedUserSimulatorConfig,
)
from google.adk.evaluation.conversation_scenarios import ConversationScenario

scenario = ConversationScenario(
    starting_prompt="I'd like to book a flight from NYC to London.",
    conversation_plan="Book a round-trip flight and confirm the booking reference.",
)

config = LlmBackedUserSimulatorConfig(
    model="gemini-2.5-flash",
    max_allowed_invocations=5,
)

sim = LlmBackedUserSimulator(config=config, conversation_scenario=scenario)
# Drives the conversation using LLM; stops when </finished> is emitted or limit hit
```

### Example 2 — overriding thinking budget for faster tests

```python
from google.adk.evaluation.simulation.llm_backed_user_simulator import (
    LlmBackedUserSimulatorConfig,
)
from google.genai import types

config = LlmBackedUserSimulatorConfig(
    model="gemini-2.5-flash",
    model_configuration=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,   # disable thinking to speed up tests
            thinking_budget=0,
        )
    ),
    max_allowed_invocations=10,
)
```

### Example 3 — custom instructions with Jinja placeholders

```python
from google.adk.evaluation.simulation.llm_backed_user_simulator import (
    LlmBackedUserSimulatorConfig,
)

config = LlmBackedUserSimulatorConfig(
    model="gemini-2.5-flash",
    custom_instructions="""You are an impatient customer.
Ask your questions quickly and bluntly.
Conversation so far:
{{ conversation_history }}
When the task is complete, output {{ stop_signal }}.
Your goal: {{ conversation_plan }}
""",
    max_allowed_invocations=8,
)
# The field_validator requires {{ stop_signal }}, {{ conversation_plan }},
# AND {{ conversation_history }} — omitting any one raises ValueError.
# {{ stop_signal }} → </finished>; {{ conversation_plan }} → scenario plan
```

---

## 6 · `CachePerformanceAnalyzer` — context-cache analytics

**Source:** `google/adk/utils/cache_performance_analyzer.py`

`CachePerformanceAnalyzer` reads the event history of a session from a `BaseSessionService` and computes cache utilisation metrics. All methods are async since session lookup is I/O bound.

### Key implementation details

```python
@experimental
class CachePerformanceAnalyzer:
    def __init__(self, session_service: BaseSessionService): ...

    async def _get_agent_cache_history(
        self, session_id, user_id, app_name, agent_name=None
    ) -> List[CacheMetadata]:
        # Filters session.events by event.cache_metadata is not None
        # and optionally by event.author == agent_name

    async def analyze_agent_cache_performance(
        self, session_id, user_id, app_name, agent_name
    ) -> Dict[str, Any]:
        # Returns:
        #   status: "active" | "no_cache_data"
        #   requests_with_cache: len(cache_history)
        #   avg_invocations_used: mean of c.invocations_used across history
        #   latest_cache: cache_history[-1].cache_name
        #   cache_refreshes: len({c.cache_name for c in cache_history})
        #   total_invocations: sum(invocations_used)
        #   total_prompt_tokens, total_cached_tokens
        #   cache_hit_ratio_percent = (cached / prompt) * 100
        #   cache_utilization_ratio_percent = (hits / total_requests) * 100
        #   avg_cached_tokens_per_request = cached / total_requests
        #   total_requests, requests_with_cache_hits
```

`cache_refreshes` counts distinct cache names — each new cache version (created after `cache_intervals` invocations or TTL expiry) bumps this counter. A low `avg_invocations_used` suggests the cache is being discarded before it pays off.

### Example 1 — basic performance report

```python
import asyncio
from google.adk.utils.cache_performance_analyzer import CachePerformanceAnalyzer
from google.adk.sessions import InMemorySessionService

session_service = InMemorySessionService()

async def report():
    # create_session must be called first; get_session() returns None for unknown IDs
    await session_service.create_session(
        app_name="my-app", user_id="user-1", session_id="session-123"
    )
    analyzer = CachePerformanceAnalyzer(session_service)
    result = await analyzer.analyze_agent_cache_performance(
        session_id="session-123",
        user_id="user-1",
        app_name="my-app",
        agent_name="research_agent",
    )
    if result["status"] == "no_cache_data":
        print("No caching activity recorded yet.")
    else:
        print(f"Cache hit ratio: {result['cache_hit_ratio_percent']:.1f}%")
        print(f"Cache utilisation: {result['cache_utilization_ratio_percent']:.1f}%")
        print(f"Avg cached tokens/request: {result['avg_cached_tokens_per_request']:.0f}")
```

### Example 2 — detecting cache churn

```python
async def check_churn(analyzer, session_id, user_id, app_name, agent_name):
    r = await analyzer.analyze_agent_cache_performance(
        session_id, user_id, app_name, agent_name
    )
    if r["status"] == "no_cache_data":
        return
    # A large number of cache_refreshes relative to requests_with_cache
    # indicates the cache expires before it covers enough requests.
    churn_rate = r["cache_refreshes"] / max(r["requests_with_cache"], 1)
    if churn_rate > 0.5:
        print(f"High churn: {r['cache_refreshes']} refreshes for "
              f"{r['requests_with_cache']} cached requests. "
              "Consider increasing ttl_seconds in ContextCacheConfig.")
```

### Example 3 — no-cache path guard

```python
async def safe_report(analyzer, **kwargs):
    r = await analyzer.analyze_agent_cache_performance(**kwargs)
    match r.get("status"):
        case "no_cache_data":
            print("Agent not using context caching yet.")
        case "active":
            print(f"Latest cache: {r['latest_cache']}")
            print(f"Total invocations: {r['total_invocations']}")
        case _:
            print(f"Unknown status: {r['status']}")
```

---

## 7 · `ContextCacheConfig` — app-level context caching configuration

**Source:** `google/adk/agents/context_cache_config.py`

`ContextCacheConfig` is attached to an `App` to enable and tune context caching for all `LlmAgent` instances in the app. When the field is `None` on the app, caching is disabled entirely.

### Key implementation details

```python
@experimental(FeatureName.AGENT_CONFIG)
class ContextCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_intervals: int = Field(default=10, ge=1, le=100,
        description="Max invocations to reuse the same cache before refresh")

    ttl_seconds: int = Field(default=1800, gt=0,
        description="Time-to-live for the cache in seconds (default 30 min)")

    min_tokens: int = Field(default=0, ge=0,
        description=(
            "Minimum prior-request tokens required to enable caching. "
            "Gemini enforces a hard 4096-token minimum that always applies, "
            "so values below 4096 have no additional effect. "
            "No cache on the first request of a session (gate on second turn)."
        ))

    create_http_options: types.HttpOptions | None = Field(default=None,
        description=(
            "HTTP options for CachedContent.create() calls. "
            "Use types.HttpOptions(timeout=10000) for a 10-second timeout."
        ))

    @property
    def ttl_string(self) -> str:
        return f"{self.ttl_seconds}s"   # e.g. "1800s"

    def __str__(self) -> str:
        return (f"ContextCacheConfig(cache_intervals={self.cache_intervals}, "
                f"ttl={self.ttl_seconds}s, min_tokens={self.min_tokens}, "
                f"create_http_options={self.create_http_options})")
```

**Caching invariants (from source docstring):**
- Caching begins on the **second turn at the earliest** — the first request has no prior token count to gate on.
- Gemini enforces a **hard 4096-token minimum** regardless of `min_tokens`.
- A cache is refreshed after `cache_intervals` invocations or when the TTL expires.
- When `create_http_options` timeout fires during cache creation, the request proceeds without a cache — no error is raised.

### Example 1 — enabling caching on an App

```python
from google.adk.apps.app import App
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.agents.llm_agent import LlmAgent

agent = LlmAgent(
    name="summariser",
    model="gemini-2.5-flash",
    instruction="Summarise the given document.",
)

app = App(
    name="summariser-app",
    root_agent=agent,
    context_cache_config=ContextCacheConfig(
        cache_intervals=5,     # refresh cache every 5 invocations
        ttl_seconds=3600,      # 1 hour TTL
        min_tokens=8192,       # only cache if prior request had >8192 tokens
    ),
)
print(app.context_cache_config.ttl_string)   # "3600s"
```

### Example 2 — adding a timeout on cache creation

```python
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.genai import types

config = ContextCacheConfig(
    cache_intervals=10,
    ttl_seconds=1800,
    create_http_options=types.HttpOptions(timeout=5000),  # 5-second timeout
    # If CachedContent.create() exceeds 5s, request proceeds uncached
)
print(str(config))
# ContextCacheConfig(cache_intervals=10, ttl=1800s, min_tokens=0,
#   create_http_options=HttpOptions(timeout=5000))
```

### Example 3 — disabling caching selectively

```python
from google.adk.apps.app import App
from google.adk.agents.llm_agent import LlmAgent

# No context_cache_config → caching disabled for this app
app_no_cache = App(
    name="quick-app",
    root_agent=LlmAgent(name="q", model="gemini-2.0-flash"),
    # context_cache_config=None  (default)
)
print(app_no_cache.context_cache_config)   # None
```

---

## 8 · `TelemetryContext` + `start_as_current_node_span` — workflow node telemetry

**Source:** `google/adk/telemetry/node_tracing.py`

`TelemetryContext` is a frozen dataclass that pairs an OTel context with a list of event IDs emitted within a node span. `start_as_current_node_span` is the async context manager that opens the correct span type based on what kind of node is running.

### Key implementation details

```python
GEN_AI_WORKFLOW_NESTED = "gen_ai.workflow.nested"
_ENTRYPOINT_WORKFLOW_KEY = context_api.create_key("adk-entrypoint-workflow-active")

@dataclass(frozen=True)
class TelemetryContext:
    otel_context: context_api.Context
    _associated_event_ids: list[str] = field(default_factory=list)

    def add_event(self, event: Event) -> None:
        self._associated_event_ids.append(event.id)

@asynccontextmanager
async def start_as_current_node_span(
    context: Context, node: BaseNode
) -> AsyncIterator[TelemetryContext]:
    # Dispatches to:
    #   BaseAgent → _invoke_agent_span  (passes through; agent emits its own span)
    #   Workflow  → _invoke_workflow_span  (opens invoke_workflow {name} span)
    #   BaseNode  → _invoke_node_span  (opens invoke_node {name} span)
```

**Span naming (aligned with OTel semconv):**
- `invoke_agent {agent.name}` — agents emit these themselves (semconv v1.36).
- `invoke_workflow {workflow.name}` — emitted for `Workflow` nodes (semconv v1.41).
- `invoke_node {node.name}` — for plain `BaseNode` subclasses (no semconv standard yet).

**Nested workflow detection:** `_ENTRYPOINT_WORKFLOW_KEY` is a ContextVar key set to `True` once the first workflow span opens. Subsequent workflow spans check this key and emit `gen_ai.workflow.nested = True` if already set. The root workflow never emits this attribute.

After a span closes, `_maybe_set_associated_events` stamps `gcp.vertex.agent.associated_event_ids` onto the span when any event IDs were recorded.

### Example 1 — manually tracking events within a span

```python
from google.adk.telemetry.node_tracing import TelemetryContext
from opentelemetry import context as context_api

# TelemetryContext is a frozen dataclass — add_event mutates the internal list
# (mutable list in a frozen dataclass is allowed because only the reference is frozen)
tel_ctx = TelemetryContext(otel_context=context_api.get_current())

class FakeEvent:
    id = "evt-001"

tel_ctx.add_event(FakeEvent())
print(tel_ctx._associated_event_ids)   # ['evt-001']
```

### Example 2 — nested workflow flag propagation

```python
# When two Workflow nodes are nested (e.g. an agent-as-tool spins up a sub-workflow),
# the outer workflow sets _ENTRYPOINT_WORKFLOW_KEY in the OTel context.
# The inner workflow reads the key and sets gen_ai.workflow.nested = True on its span.

# You can inspect this programmatically:
from google.adk.telemetry.node_tracing import _ENTRYPOINT_WORKFLOW_KEY
from opentelemetry import context as context_api

ctx = context_api.get_current()
is_nested = bool(context_api.get_value(_ENTRYPOINT_WORKFLOW_KEY, ctx))
print(f"Running inside a nested workflow: {is_nested}")
```

### Example 3 — reading associated events from a span attribute

```python
# After a node completes, its span carries associated_event_ids.
# In a custom OTel span processor you can extract them:

from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace import ReadableSpan

class EventIdLogger(SpanProcessor):
    def on_end(self, span: ReadableSpan) -> None:
        ids = span.attributes.get("gcp.vertex.agent.associated_event_ids")
        if ids:
            print(f"Span '{span.name}' produced events: {list(ids)}")
```

---

## 9 · `JoinNode` + `Trigger` — fan-in barrier and trigger model

**Source:** `google/adk/workflow/_join_node.py`, `google/adk/workflow/_trigger.py`

`JoinNode` is a workflow barrier that waits for **all predecessor branches** to fire before outputting. `Trigger` is the pydantic data model that carries a pending activation from one node to the next.

### Key implementation details

```python
class Trigger(BaseModel):
    model_config = ConfigDict(ser_json_bytes='base64')  # bytes serialised as base64
    input: Any = None                   # payload forwarded to the triggered node
    use_sub_branch: bool = False        # activate sub-branch scoping
    branch: str | None = None          # branch tag from predecessor
    isolation_scope: str | None = None  # explicit scope tag propagated downstream

class JoinNode(BaseNode):
    @property
    def _requires_all_predecessors(self) -> bool:
        return True    # orchestrator holds output until every predecessor fires

    def _validate_input_data(self, data: Any) -> Any:
        # When input_schema is set and data is a dict:
        # validates each value individually → {key: validated_value}
        if self.input_schema and isinstance(data, dict):
            return {k: self._validate_schema(v, self.input_schema) for k, v in data.items()}
        return super()._validate_input_data(data)

    async def _run_impl(self, *, ctx: Context, node_input: Any):
        # node_input is the aggregated dict of all predecessor outputs
        yield Event(output=node_input, branch=ctx._invocation_context.branch)
```

`JoinNode` does no processing of its own; it simply yields the aggregated `node_input` (a dict of predecessor outputs keyed by branch or node name) as an event. This is the canonical fan-in pattern after a parallel fan-out.

### Example 1 — parallel fan-out into a JoinNode

```python
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._node import node
from google.adk.workflow._join_node import JoinNode

@node
async def branch_a(ctx):
    yield {"result_a": "done A"}

@node
async def branch_b(ctx):
    yield {"result_b": "done B"}

join = JoinNode(name="merge")

wf = Workflow(
    name="fan_out_fan_in",
    edges=[
        ("START", (branch_a, branch_b)),   # fan-out
        (branch_a, join),
        (branch_b, join),                  # both must complete before join fires
    ],
)
```

### Example 2 — inspecting Trigger fields for sub-branch routing

```python
from google.adk.workflow._trigger import Trigger

# A trigger carrying a payload for a sub-branch
t = Trigger(
    input={"query": "What is ADK?"},
    use_sub_branch=True,
    branch="main.search",
    isolation_scope="search_scope",
)
print(t.input)           # {'query': 'What is ADK?'}
print(t.use_sub_branch)  # True
print(t.branch)          # main.search
print(t.isolation_scope) # search_scope

# Serialise to JSON (bytes fields would be base64-encoded)
import json
print(json.loads(t.model_dump_json()))
```

### Example 3 — JoinNode with schema validation

```python
from pydantic import BaseModel
from google.adk.workflow._join_node import JoinNode

class BranchResult(BaseModel):
    value: int
    label: str

join = JoinNode(name="validated_join", input_schema=BranchResult)
# When node_input is {"branch_a": {"value": 42, "label": "A"},
#                      "branch_b": {"value": 7,  "label": "B"}},
# _validate_input_data validates each sub-dict against BranchResult.
```

---

## 10 · `Graph` + `Edge` + `RouteValue` / `RoutingMap` / `ChainElement` / `EdgeItem` — workflow graph primitives

**Source:** `google/adk/workflow/_graph.py`

`Graph` is the internal data model that `Workflow` uses to store nodes and edges. `Edge` defines a directed connection with optional route values for conditional branching. The type aliases (`RouteValue`, `NodeLike`, `RoutingMap`, `ChainElement`, `EdgeItem`) are the building blocks of the fluent chain syntax.

### Key implementation details

```python
RouteValue: TypeAlias = bool | int | str
NodeLike: TypeAlias = BaseNode | BaseTool | Callable[..., Any] | Literal["START"]
RoutingMap: TypeAlias = dict[RouteValue, NodeLike | tuple[NodeLike, ...]]
ChainElement: TypeAlias = NodeLike | tuple[NodeLike, ...] | RoutingMap
EdgeItem: TypeAlias = Edge | tuple[ChainElement, ...]
DEFAULT_ROUTE = "__DEFAULT__"

class Edge(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    from_node: Annotated[BaseNode, SerializeAsAny()]
    to_node: Annotated[BaseNode, SerializeAsAny()]
    route: RouteValue | list[RouteValue] | None = None

class Graph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    nodes: list[BaseNode] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    _terminal_node_names: set[str] = PrivateAttr(default_factory=set)

    @classmethod
    def from_edge_items(cls, edge_items: list[EdgeItem]) -> Graph: ...

    def model_post_init(self, context):
        # Raises if nodes set explicitly (inferred from edges)
        # Deduplicates by id(node) — preserves insertion order
        nodes = {id(n): n for e in self.edges for n in [e.from_node, e.to_node]}
        self.nodes = list(nodes.values())

    def get_next_pending_nodes(
        self, node_name: str, routes_to_match: RouteValue | list[RouteValue] | None
    ) -> list[str]:
        # 1. Edges with route=None → always triggered
        # 2. Edges matching any value in routes_to_match → triggered; sets matched flag
        # 3. If no match and DEFAULT_ROUTE edge exists → triggered
        # 4. If has_routing_edges and still empty → logs a warning; branch ends silently
```

`Graph.get_next_pending_nodes()` is the routing engine: given the current node name and the route(s) it emitted, it returns the list of successor node names to move to `PENDING` state. `DEFAULT_ROUTE = "__DEFAULT__"` is an **edge-side tag**: when you define an edge with `route=DEFAULT_ROUTE`, that edge fires as a fallback only when no other edge's route matched what the node emitted. It is not a value to emit from the node itself — emitting `DEFAULT_ROUTE` as a route value will not reliably trigger the fallback and may silently end the branch.

### Example 1 — `Graph.from_edge_items` with a tuple chain

```python
from google.adk.workflow._graph import Graph, Edge
from google.adk.workflow._node import node

@node
async def step1(ctx): yield "step1"
@node
async def step2(ctx): yield "step2"
@node
async def step3(ctx): yield "step3"

# Tuple chain: (a, b, c) → a→b, b→c
g = Graph.from_edge_items([(step1, step2, step3)])
print([n.name for n in g.nodes])   # ['step1', 'step2', 'step3']
print([(e.from_node.name, e.to_node.name) for e in g.edges])
# [('step1', 'step2'), ('step2', 'step3')]
```

### Example 2 — conditional routing with DEFAULT fallback

```python
from google.adk.workflow._graph import Graph, DEFAULT_ROUTE
from google.adk.workflow._node import node

@node
async def router(ctx):
    ctx.route = "fast"   # route is a property setter, not a callable
    return               # explicit return; routing is controlled via ctx.route, not yielded output

@node
async def fast_path(ctx): yield "fast result"
@node
async def slow_path(ctx): yield "slow result"

# RoutingMap syntax: route → target node
g = Graph.from_edge_items([
    (router, {"fast": fast_path, DEFAULT_ROUTE: slow_path}),
])

# When router emits "fast":
next_nodes = g.get_next_pending_nodes("router", "fast")
print(next_nodes)   # ['fast_path']

# When router emits an unrecognised value:
next_nodes = g.get_next_pending_nodes("router", "unknown")
print(next_nodes)   # ['slow_path']  (DEFAULT_ROUTE fallback)
```

### Example 3 — fan-out with a RoutingMap

```python
from google.adk.workflow._graph import Graph
from google.adk.workflow._node import node

@node
async def dispatcher(ctx): yield "go"
@node
async def worker_a(ctx): yield "a"
@node
async def worker_b(ctx): yield "b"

# Fan-out to both workers when route is "go"
g = Graph.from_edge_items([
    (dispatcher, {"go": (worker_a, worker_b)}),
])

next_nodes = g.get_next_pending_nodes("dispatcher", "go")
print(next_nodes)   # ['worker_a', 'worker_b']

# nodes are auto-inferred from edges — no need to declare them explicitly
print([n.name for n in g.nodes])
# ['dispatcher', 'worker_a', 'worker_b']
```

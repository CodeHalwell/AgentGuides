---
title: "Class deep dives — volume 19 (tools, retrieval, and workflow patterns)"
description: "Source-verified 2.2.0 deep dives: TransferToAgentTool (enum-constrained agent transfer), SetModelResponseTool (output_schema + tools workaround), PreloadMemoryTool (automatic memory injection), VertexAiSearchTool (built-in Vertex AI Search with subclassing), FilesRetrieval (local directory RAG via LlamaIndex), ToolboxToolset (MCP Toolbox delegate), JoinNode (fan-in synchronization), @node decorator and FunctionNode, _ParallelWorker (fan-out map), Workflow.max_concurrency and edge patterns."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 19"
  order: 88
---

Source-verified against **google-adk==2.2.0** (installed from PyPI, June 2026). Every field name, signature, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `TransferToAgentTool` | `google.adk.tools.transfer_to_agent_tool` | Stable |
| 2 | `SetModelResponseTool` | `google.adk.tools.set_model_response_tool` | Stable (internal) |
| 3 | `PreloadMemoryTool` | `google.adk.tools.preload_memory_tool` | Stable |
| 4 | `VertexAiSearchTool` | `google.adk.tools.vertex_ai_search_tool` | Stable |
| 5 | `FilesRetrieval` | `google.adk.tools.retrieval.files_retrieval` | Stable |
| 6 | `ToolboxToolset` | `google.adk.tools.toolbox_toolset` | Stable |
| 7 | `JoinNode` | `google.adk.workflow._join_node` | Stable |
| 8 | `@node` decorator + `FunctionNode` | `google.adk.workflow._node` | Stable |
| 9 | `_ParallelWorker` | `google.adk.workflow._parallel_worker` | Stable |
| 10 | `Workflow.max_concurrency` + edge patterns | `google.adk.workflow._workflow` | Stable |

---

## 1 · `TransferToAgentTool` — enum-constrained agent transfer

**Source:** `google.adk.tools.transfer_to_agent_tool`

The free `transfer_to_agent(agent_name, tool_context)` function lets an LLM write any string it likes into `agent_name`, which means it can hallucinate names that do not correspond to any real sub-agent. `TransferToAgentTool` is a `FunctionTool` subclass that wraps the same function but overrides `_get_declaration()` to inject a JSON Schema `enum` constraint on `agent_name`. Both `function_decl.parameters.properties["agent_name"].enum` (the `types.Schema` path) and `function_decl.parameters_json_schema["properties"]["agent_name"]["enum"]` (the dict path) are patched. The result is that the model can only output one of the explicitly listed names, preventing routing errors at the cost of one small Python list.

### Constructor (source-verified)

```python
from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

TransferToAgentTool(
    agent_names: list[str],   # the only valid agent names the model may choose
)
```

Inherits all `FunctionTool` behaviour. The `func` is always `transfer_to_agent`; you do not pass it yourself.

### Example 1 — basic three-way router

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

# mode='single_turn' keeps agents in the tree (so find_agent() can locate them
# during transfer) but excludes them from _AgentTransferLlmRequestProcessor's
# auto-generated TransferToAgentTool, avoiding duplicate tool declarations.
billing_agent = LlmAgent(
    name="billing",
    model="gemini-2.0-flash",
    mode="single_turn",
    instruction="You handle billing questions, invoices, and payment issues.",
)
support_agent = LlmAgent(
    name="support",
    model="gemini-2.0-flash",
    mode="single_turn",
    instruction="You handle technical support and troubleshooting.",
)
sales_agent = LlmAgent(
    name="sales",
    model="gemini-2.0-flash",
    mode="single_turn",
    instruction="You handle new subscriptions, upgrades, and pricing.",
)

orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.0-flash",
    instruction=(
        "You are a triage agent. Understand the user's request and transfer "
        "to the most appropriate specialist: billing for payment issues, "
        "support for technical problems, sales for new purchases."
    ),
    tools=[TransferToAgentTool(agent_names=["billing", "support", "sales"])],
    sub_agents=[billing_agent, support_agent, sales_agent],
)
```

### Example 2 — dynamic agent list built at runtime

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool


def build_orchestrator(enabled_agents: list[str]) -> LlmAgent:
    """Build an orchestrator whose transfer menu reflects runtime config."""
    # mode='single_turn' registers agents in the tree without triggering
    # the auto-TransferToAgentTool that would duplicate the explicit tool.
    sub_agents = [
        LlmAgent(
            name=name,
            model="gemini-2.0-flash",
            mode="single_turn",
            instruction=f"You are the {name} specialist agent.",
        )
        for name in enabled_agents
    ]

    return LlmAgent(
        name="router",
        model="gemini-2.0-flash",
        instruction=(
            "Route the user to the correct specialist. "
            f"Available specialists: {', '.join(enabled_agents)}."
        ),
        tools=[TransferToAgentTool(agent_names=enabled_agents)],
        sub_agents=sub_agents,
    )


# Production build: all three agents enabled
prod_router = build_orchestrator(["billing", "support", "sales"])

# Staging build: only support and sales
staging_router = build_orchestrator(["support", "sales"])
```

### Example 3 — contrasting with the raw `transfer_to_agent` function

```python
# The free function — no enum constraint, model can hallucinate any name
from google.adk.tools.transfer_to_agent_tool import transfer_to_agent
from google.adk.agents.llm_agent import LlmAgent

# BAD: LLM could output agent_name="refunds" even if that agent does not exist
unconstrained_agent = LlmAgent(
    name="bad_router",
    model="gemini-2.0-flash",
    instruction="Route to billing or support.",
    tools=[transfer_to_agent],   # free function — no schema constraints
)

# GOOD: LLM can ONLY output "billing" or "support"
from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

constrained_agent = LlmAgent(
    name="good_router",
    model="gemini-2.0-flash",
    instruction="Route to billing or support.",
    tools=[TransferToAgentTool(agent_names=["billing", "support"])],
)
```

### Example 4 — inspecting the generated FunctionDeclaration

```python
from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

tool = TransferToAgentTool(agent_names=["alpha", "beta", "gamma"])
decl = tool._get_declaration()

# Verify the enum was injected into types.Schema properties
if decl and decl.parameters and decl.parameters.properties:
    agent_name_schema = decl.parameters.properties.get("agent_name")
    print("Enum from types.Schema:", agent_name_schema.enum)
    # Enum from types.Schema: ['alpha', 'beta', 'gamma']

# Verify the enum was injected into the JSON Schema dict
if decl and decl.parameters_json_schema:
    props = decl.parameters_json_schema.get("properties", {})
    print("Enum from JSON schema:", props.get("agent_name", {}).get("enum"))
    # Enum from JSON schema: ['alpha', 'beta', 'gamma']
```

### Example 5 — multi-level routing hierarchy

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

# Leaf agents
billing_invoice = LlmAgent(
    name="billing_invoice",
    model="gemini-2.0-flash",
    instruction="Handle invoice generation and downloads.",
)
billing_payment = LlmAgent(
    name="billing_payment",
    model="gemini-2.0-flash",
    instruction="Handle payment processing and refunds.",
)

# Mid-level router
billing_router = LlmAgent(
    name="billing",
    model="gemini-2.0-flash",
    instruction="Handle all billing matters. Route to invoice or payment specialists.",
    tools=[TransferToAgentTool(agent_names=["billing_invoice", "billing_payment"])],
    sub_agents=[billing_invoice, billing_payment],
)

support_agent = LlmAgent(
    name="support",
    model="gemini-2.0-flash",
    instruction="Handle all support and troubleshooting requests.",
)

# Top-level router — only sees billing and support, not their internals
top_router = LlmAgent(
    name="top_router",
    model="gemini-2.0-flash",
    instruction="Classify the request: billing concerns go to billing, technical issues go to support.",
    tools=[TransferToAgentTool(agent_names=["billing", "support"])],
    sub_agents=[billing_router, support_agent],
)
```

---

## 2 · `SetModelResponseTool` — `output_schema` + tools workaround

**Source:** `google.adk.tools.set_model_response_tool`

Normally, setting `output_schema` on an `LlmAgent` disables all tool use — Gemini cannot produce both a structured JSON response and tool calls in the same turn. `SetModelResponseTool` is an **internal** `BaseTool` that circumvents this restriction. When the framework detects that an agent has both `output_schema` and `tools`, it auto-injects `SetModelResponseTool` and instructs the model to call `set_model_response(...)` instead of outputting JSON text. The tool's `run_async` validates the args against the output schema and writes the result to `tool_context.actions.set_model_response`.

The schema shape determines which parameter name is generated: a `BaseModel` subclass expands all model fields as individual keyword arguments; a `list[BaseModel]` generates a single `items: list[...]` parameter; any other type (`list[str]`, `dict`, etc.) generates a single `response: T` parameter.

### Constructor (source-verified)

```python
from google.adk.tools.set_model_response_tool import SetModelResponseTool

SetModelResponseTool(
    output_schema: SchemaType,
    # SchemaType is a union of:
    #   type[BaseModel]         — e.g. MySchema
    #   list[type[BaseModel]]   — e.g. list[MySchema]  (uses 'items' param)
    #   list[primitive]         — e.g. list[str], list[int]  (uses 'response' param)
    #   dict                    — raw JSON schema dict       (uses 'response' param)
    #   types.Schema            — Google genai Schema        (uses 'response' param)
)
```

You never instantiate this yourself. Use `output_schema` on `LlmAgent` alongside `tools` and the framework handles injection automatically.

### Example 1 — enabling the workaround via `LlmAgent` fields

```python
from pydantic import BaseModel
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.google_search_tool import google_search


class ResearchSummary(BaseModel):
    title: str
    key_findings: list[str]
    confidence: float


# Normally output_schema disables tools — but not when tools are also present.
# The framework calls _should_use_set_model_response_tool() and, when True,
# injects SetModelResponseTool automatically.
research_agent = LlmAgent(
    name="researcher",
    model="gemini-2.0-flash",
    instruction=(
        "Research the topic using Google Search, then call set_model_response "
        "with your findings structured as title, key_findings, and confidence."
    ),
    tools=[google_search],          # tool use enabled
    output_schema=ResearchSummary,  # structured output required
)
```

### Example 2 — `list[BaseModel]` output with tools

```python
from pydantic import BaseModel
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.google_search_tool import google_search


class Article(BaseModel):
    url: str
    title: str
    snippet: str


# list[BaseModel] → set_model_response has a single parameter: items: list[Article]
article_collector = LlmAgent(
    name="article_collector",
    model="gemini-2.0-flash",
    instruction=(
        "Search for articles on the topic and call set_model_response "
        "with a list of Article objects (url, title, snippet)."
    ),
    tools=[google_search],
    output_schema=list[Article],
)
```

### Example 3 — primitive list output with tools

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.google_search_tool import google_search


# list[str] → set_model_response has a single parameter: response: list[str]
tag_extractor = LlmAgent(
    name="tag_extractor",
    model="gemini-2.0-flash",
    instruction=(
        "Search for the topic and call set_model_response with a list of "
        "relevant keyword tags as strings."
    ),
    tools=[google_search],
    output_schema=list[str],
)
```

### Example 4 — understanding `run_async` validation behaviour

```python
# This example shows what SetModelResponseTool.run_async() does internally.
# You would not call this in production code — it is called by the framework.
from pydantic import BaseModel
from google.adk.tools.set_model_response_tool import SetModelResponseTool


class Sentiment(BaseModel):
    label: str
    score: float


tool = SetModelResponseTool(output_schema=Sentiment)

# The tool expects keyword args matching model fields:
# { "label": "positive", "score": 0.92 }
# run_async() calls Sentiment.model_validate(args), then model_dump(exclude_none=True),
# then writes result to tool_context.actions.set_model_response.

# For list[BaseModel], args are { "items": [{"label": "pos", "score": 0.9}, ...] }
list_tool = SetModelResponseTool(output_schema=list[Sentiment])

# For list[str], args are { "response": ["tag1", "tag2"] }
str_list_tool = SetModelResponseTool(output_schema=list[str])

print(tool.name)          # "set_model_response"
print(list_tool.name)     # "set_model_response"
print(str_list_tool.name) # "set_model_response"
```

### Example 5 — BaseModel schema combined with tools

```python
from pydantic import BaseModel
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.google_search_tool import google_search


class SearchSummary(BaseModel):
    summary: str
    sources: list[str]


# Use a BaseModel — not a raw dict — when combining output_schema with tools.
# SetModelResponseTool assigns the dict as a Python type annotation, which
# fails during FunctionDeclaration building; BaseModel is the supported path.
search_agent = LlmAgent(
    name="search_agent",
    model="gemini-2.0-flash",
    instruction=(
        "Search for information and call set_model_response with "
        "a summary string and list of source URLs."
    ),
    tools=[google_search],
    output_schema=SearchSummary,
)
```

---

## 3 · `PreloadMemoryTool` — automatic memory injection

**Source:** `google.adk.tools.preload_memory_tool`

`PreloadMemoryTool` is a `BaseTool` subclass that **only** overrides `process_llm_request` — it never implements `run_async` and is never invoked by the model. On every LLM request the framework calls `process_llm_request`, which searches the memory service for the current user query and injects any matching past conversations into the system instruction as a `<PAST_CONVERSATIONS>` XML block. The memory entries include timestamps (`Time: ...`) and author prefixes when available. If the search fails, the tool logs a warning and continues silently. The module exports a singleton `preload_memory_tool` instance — import that directly rather than constructing a new one.

### Constructor (source-verified)

```python
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

PreloadMemoryTool()
# name="preload_memory", description="preload_memory" — neither is user-visible.
# The singleton is exported as:
from google.adk.tools.preload_memory_tool import preload_memory_tool
```

The tool reads `tool_context.user_content.parts[0].text` as the search query. Only text parts are extracted from memory entries.

### Example 1 — adding `preload_memory_tool` to an agent

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.preload_memory_tool import preload_memory_tool
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService

# Memory service must be configured on Runner; App does not accept it.
memory_service = VertexAiMemoryBankService(
    project="my-gcp-project",
    location="us-central1",
    agent_engine_id="123",   # numeric ID only, not the full resource path
)

agent = LlmAgent(
    name="memory_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You are a helpful assistant with access to past conversations. "
        "Use them to provide continuity across sessions."
    ),
    tools=[preload_memory_tool],   # injected before every LLM call
)

runner = Runner(
    app_name="memory_app",
    agent=agent,
    session_service=InMemorySessionService(),
    memory_service=memory_service,
)
```

### Example 2 — combining with `add_session_to_memory` for full memory cycle

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.preload_memory_tool import preload_memory_tool
from google.adk.agents.callback_context import CallbackContext


# ADK calls this as callback(callback_context=...) so the parameter name must match.
async def save_memory_after_turn(callback_context: CallbackContext) -> None:
    """Callback: persist session to long-term memory after every agent turn."""
    await callback_context.add_session_to_memory()


agent = LlmAgent(
    name="persistent_agent",
    model="gemini-2.0-flash",
    instruction="You remember the user across sessions.",
    tools=[preload_memory_tool],
    after_agent_callback=save_memory_after_turn,
)
```

### Example 3 — what `process_llm_request` injects

```python
# For clarity, here is the exact text that PreloadMemoryTool injects when
# a memory search returns results. You do not call this directly.

INJECTED_SYSTEM_INSTRUCTION_TEMPLATE = """\
The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
{memory_text}
</PAST_CONVERSATIONS>
"""

# memory_text is built as:
# For each MemoryEntry:
#   if entry.timestamp: "Time: <timestamp>"
#   if entry.author:    "<author>: <text>"
#   else:               "<text>"
# All lines joined with "\n"

# Example injected content:
example = """\
The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
Time: 2026-06-10T14:23:00Z
user: What's the capital of France?
assistant: The capital of France is Paris.
</PAST_CONVERSATIONS>
"""
```

### Example 4 — using `InMemoryMemoryService` for local testing

```python
import asyncio
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.preload_memory_tool import preload_memory_tool
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types


async def main():
    memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()

    agent = LlmAgent(
        name="test_memory_agent",
        model="gemini-2.0-flash",
        instruction="Answer concisely. Recall past conversations if relevant.",
        tools=[preload_memory_tool],
    )

    runner = Runner(
        agent=agent,
        app_name="test_app",
        session_service=session_service,
        memory_service=memory_service,
    )

    session = await session_service.create_session(app_name="test_app", user_id="u1")
    msg = types.Content(role="user", parts=[types.Part.from_text(text="My name is Alice.")])

    async for event in runner.run_async(
        user_id="u1", session_id=session.id, new_message=msg
    ):
        if event.is_final_response() and event.content:
            print("".join(p.text or "" for p in event.content.parts))


asyncio.run(main())
```

### Example 5 — graceful degradation when memory search fails

```python
# PreloadMemoryTool wraps search_memory() in try/except — if the memory
# service is unreachable, it logs a warning and returns None, so the agent
# continues without memory context rather than crashing.
#
# To observe this behaviour, point at a non-existent memory service:

import logging
logging.basicConfig(level=logging.WARNING)

from google.adk.tools.preload_memory_tool import preload_memory_tool

# The tool's process_llm_request contains:
#   try:
#       response = await tool_context.search_memory(user_query)
#   except Exception:
#       logging.warning('Failed to preload memory for query: %s', user_query)
#       return  # ← silently skips injection, agent still runs

# No extra configuration needed — degradation is automatic.
print(f"Tool name: {preload_memory_tool.name}")   # preload_memory
```

---

## 4 · `VertexAiSearchTool` — built-in Vertex AI Search with subclassing

**Source:** `google.adk.tools.vertex_ai_search_tool`

`VertexAiSearchTool` is a `BaseTool` that works by overriding `process_llm_request` — it never has a user-visible `run_async`. On each LLM call it appends a `types.Tool(retrieval=types.Retrieval(vertex_ai_search=...))` object to `llm_request.config.tools`, turning on native Gemini grounding via Vertex AI Search. The constructor enforces a mutual-exclusion constraint: you must provide **either** `data_store_id` **or** `search_engine_id`, not both. If `data_store_specs` is provided, `search_engine_id` is also required.

For Gemini 1.x models, mixing `VertexAiSearchTool` with any other tool raises a `ValueError` (use `bypass_multi_tools_limit=True` to override). Gemini 2.x does not have this restriction. To apply dynamic filters (e.g., per-user scoping), subclass and override `_build_vertex_ai_search_config`.

### Constructor (source-verified)

```python
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool

VertexAiSearchTool(
    *,
    data_store_id: str | None = None,
    # Full resource path:
    # "projects/{project}/locations/{location}/collections/{collection}/dataStores/{dataStore}"

    data_store_specs: list[types.VertexAISearchDataStoreSpec] | None = None,
    # Only valid when search_engine_id is set.

    search_engine_id: str | None = None,
    # Full resource path:
    # "projects/{project}/locations/{location}/collections/{collection}/engines/{engine}"

    filter: str | None = None,
    # Filter expression applied to search results.

    max_results: int | None = None,
    # Maximum number of grounding results to return.

    bypass_multi_tools_limit: bool = False,
    # Set True to allow use alongside other tools on Gemini 1.x.
)
```

Raises `ValueError` if neither or both of `data_store_id` / `search_engine_id` are specified, or if `data_store_specs` is set without `search_engine_id`.

### Example 1 — minimal data-store search

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool

PROJECT = "my-gcp-project"
LOCATION = "us-central1"
DATA_STORE = f"projects/{PROJECT}/locations/{LOCATION}/collections/default_collection/dataStores/my-docs-store"

agent = LlmAgent(
    name="doc_search_agent",
    model="gemini-2.0-flash",
    instruction="Answer questions about our product documentation using the search tool.",
    tools=[
        VertexAiSearchTool(data_store_id=DATA_STORE),
    ],
)
```

### Example 2 — search engine with multiple data store specs

```python
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool

PROJECT = "my-gcp-project"
LOCATION = "global"
ENGINE = f"projects/{PROJECT}/locations/{LOCATION}/collections/default_collection/engines/my-search-engine"
DS_A = f"projects/{PROJECT}/locations/{LOCATION}/collections/default_collection/dataStores/store-a"
DS_B = f"projects/{PROJECT}/locations/{LOCATION}/collections/default_collection/dataStores/store-b"

agent = LlmAgent(
    name="multi_store_agent",
    model="gemini-2.0-flash",
    instruction=(
        "Answer questions by searching across our internal knowledge bases "
        "and product catalogue simultaneously."
    ),
    tools=[
        VertexAiSearchTool(
            search_engine_id=ENGINE,
            data_store_specs=[
                types.VertexAISearchDataStoreSpec(data_store=DS_A),
                types.VertexAISearchDataStoreSpec(data_store=DS_B),
            ],
            max_results=10,
        )
    ],
)
```

### Example 3 — static filter to scope results

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool

PROJECT = "my-gcp-project"
DATA_STORE = (
    f"projects/{PROJECT}/locations/us-central1"
    "/collections/default_collection/dataStores/articles-store"
)

# Only surface articles tagged "public" and authored after 2024
agent = LlmAgent(
    name="filtered_search_agent",
    model="gemini-2.0-flash",
    instruction="Only answer using public articles from 2025 onwards.",
    tools=[
        VertexAiSearchTool(
            data_store_id=DATA_STORE,
            filter='category: "public" AND publish_year >= 2025',
            max_results=5,
        )
    ],
)
```

### Example 4 — subclass with dynamic per-user filter

```python
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool
from google.adk.agents.llm_agent import LlmAgent
from google.genai import types

PROJECT = "my-gcp-project"
DATA_STORE = (
    f"projects/{PROJECT}/locations/us-central1"
    "/collections/default_collection/dataStores/user-docs"
)


class UserScopedSearchTool(VertexAiSearchTool):
    """Override _build_vertex_ai_search_config to apply a per-user filter."""

    def _build_vertex_ai_search_config(
        self, readonly_context: ReadonlyContext
    ) -> types.VertexAISearch:
        # Pull user_id from session state set at session creation
        user_id = readonly_context.state.get("user_id", "anonymous")
        return types.VertexAISearch(
            datastore=self.data_store_id,
            filter=f'owner_id = "{user_id}"',
            max_results=self.max_results or 5,
        )


agent = LlmAgent(
    name="personal_doc_agent",
    model="gemini-2.0-flash",
    instruction="Search only documents that belong to the current user.",
    tools=[UserScopedSearchTool(data_store_id=DATA_STORE, max_results=5)],
)
```

### Example 5 — `bypass_multi_tools_limit` for Gemini 1.x

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool

PROJECT = "my-gcp-project"
DATA_STORE = (
    f"projects/{PROJECT}/locations/us-central1"
    "/collections/default_collection/dataStores/my-store"
)


async def lookup_metadata(doc_id: str) -> dict:
    """Return document metadata from an internal database.

    Args:
        doc_id: Document identifier to look up.
    """
    return {"doc_id": doc_id, "author": "Alice", "version": "3.1"}


# Without bypass_multi_tools_limit=True this raises ValueError on Gemini 1.x:
# "Vertex AI search tool cannot be used with other tools in Gemini 1.x."
agent = LlmAgent(
    name="hybrid_agent",
    model="gemini-1.5-pro",
    instruction=(
        "Search for relevant documents, then call lookup_metadata "
        "to enrich results with author and version info."
    ),
    tools=[
        VertexAiSearchTool(
            data_store_id=DATA_STORE,
            bypass_multi_tools_limit=True,   # required for Gemini 1.x + other tools
        ),
        lookup_metadata,
    ],
)
```

---

## 5 · `FilesRetrieval` — local directory RAG via LlamaIndex

**Source:** `google.adk.tools.retrieval.files_retrieval`

`FilesRetrieval` is a `LlamaIndexRetrieval` subclass that builds a vector-store index from **every file in a local directory** and exposes it as an ADK tool. At construction time it calls `SimpleDirectoryReader(input_dir).load_data()` and `VectorStoreIndex.from_documents(...)` — both are synchronous and run in the constructor, so indexing happens once at startup. The default embedding model is `GoogleGenAIEmbedding` configured with `gemini-embedding-2-preview` and `embed_batch_size=1`. You can substitute any LlamaIndex-compatible `BaseEmbedding` via the `embedding_model` parameter.

Requires `pip install google-adk[llama-index]` and `pip install llama-index-embeddings-google-genai`.

### Constructor (source-verified)

```python
from google.adk.tools.retrieval.files_retrieval import FilesRetrieval

FilesRetrieval(
    *,
    name: str,            # tool name exposed to the LLM
    description: str,     # tool description for the LLM
    input_dir: str,       # directory path — all files are indexed at init time
    embedding_model: BaseEmbedding | None = None,
    # None → GoogleGenAIEmbedding("gemini-embedding-2-preview", embed_batch_size=1)
    # Any llama_index.core.base.embeddings.base.BaseEmbedding works.
)
```

Logs `INFO: Loading data from {input_dir}` during construction.

### Example 1 — indexing a local docs folder

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.retrieval.files_retrieval import FilesRetrieval

# Index all files in /data/docs at startup; the model calls this tool by name.
docs_retrieval = FilesRetrieval(
    name="search_docs",
    description=(
        "Search the internal documentation files for relevant information. "
        "Use this before answering any technical question."
    ),
    input_dir="/data/docs",
)

agent = LlmAgent(
    name="docs_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You are a documentation assistant. Always call search_docs to find "
        "relevant content before answering."
    ),
    tools=[docs_retrieval],
)
```

### Example 2 — custom embedding model

```python
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.retrieval.files_retrieval import FilesRetrieval

# Use a different Google embedding model with larger batch size
custom_embed = GoogleGenAIEmbedding(
    model_name="text-embedding-005",
    embed_batch_size=10,
)

retrieval_tool = FilesRetrieval(
    name="policy_search",
    description="Search company policy documents.",
    input_dir="/var/policies",
    embedding_model=custom_embed,
)

agent = LlmAgent(
    name="policy_agent",
    model="gemini-2.0-flash",
    instruction="Answer policy questions by searching the policy documents.",
    tools=[retrieval_tool],
)
```

### Example 3 — multiple `FilesRetrieval` tools covering different directories

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.retrieval.files_retrieval import FilesRetrieval

legal_retrieval = FilesRetrieval(
    name="search_legal",
    description="Search legal contracts and compliance documents.",
    input_dir="/data/legal",
)

technical_retrieval = FilesRetrieval(
    name="search_technical",
    description="Search engineering specifications and API references.",
    input_dir="/data/technical",
)

agent = LlmAgent(
    name="knowledge_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You have access to two document collections. "
        "Use search_legal for compliance and contract questions; "
        "use search_technical for engineering and API questions."
    ),
    tools=[legal_retrieval, technical_retrieval],
)
```

### Example 4 — lazy indexing pattern to defer startup cost

```python
import asyncio
from google.adk.tools.retrieval.files_retrieval import FilesRetrieval
from google.adk.agents.llm_agent import LlmAgent


def build_agent(docs_dir: str) -> LlmAgent:
    """Construct the agent (and index the docs) only when first called."""
    # VectorStoreIndex is built here — potentially slow on large directories.
    retrieval = FilesRetrieval(
        name="search_knowledge_base",
        description="Search the knowledge base for relevant content.",
        input_dir=docs_dir,
    )
    return LlmAgent(
        name="kb_agent",
        model="gemini-2.0-flash",
        instruction="Answer questions using the knowledge base.",
        tools=[retrieval],
    )


async def main():
    # Build and cache the agent on first request
    agent = build_agent("/data/kb")
    print(f"Agent '{agent.name}' ready with retrieval tool.")


asyncio.run(main())
```

### Example 5 — checking install requirements

```python
# FilesRetrieval raises ImportError at construction if dependencies are missing.
# Wrap in a try/except to give a clear error message.

try:
    from google.adk.tools.retrieval.files_retrieval import FilesRetrieval

    retrieval = FilesRetrieval(
        name="search_files",
        description="Search local files.",
        input_dir="/data/files",
    )
    print("FilesRetrieval initialised successfully.")

except ImportError as e:
    print(
        f"Missing dependency: {e}\n"
        "Run: pip install google-adk[llama-index] "
        "llama-index-embeddings-google-genai"
    )
```

---

## 6 · `ToolboxToolset` — MCP Toolbox delegate

**Source:** `google.adk.tools.toolbox_toolset`

`ToolboxToolset` is a thin `BaseToolset` wrapper around the `toolbox_adk.ToolboxToolset` from the `toolbox-adk` package. It delegates all work — tool discovery, auth token injection, parameter binding, and cleanup — to the underlying `RealToolboxToolset`. The ADK wrapper exists to provide a consistent `BaseToolset` interface and a clear `ImportError` when the optional dependency is missing.

The constructor accepts a positional `server_url` string and several optional keyword arguments. If both `toolset_name` and `tool_names` are omitted, **all tools** from the server are loaded. If both are provided, the resulting set is the union of tools from both selectors.

### Constructor (source-verified)

```python
from google.adk.tools.toolbox_toolset import ToolboxToolset

ToolboxToolset(
    server_url: str,
    # URL of the MCP Toolbox server, e.g. "http://127.0.0.1:5000"

    toolset_name: str | None = None,
    # Load all tools belonging to this named toolset on the server.

    tool_names: list[str] | None = None,
    # Load only these specific tool names.

    auth_token_getters: Mapping[str, Callable[[], str]] | None = None,
    # Map of auth-service-name → callable returning a bearer token.

    bound_params: Mapping[str, Callable[[], Any] | Any] | None = None,
    # Pre-bind parameter values (static or dynamic via callable).

    credentials: CredentialConfig | None = None,
    # toolbox_adk.CredentialConfig object for service-account auth.

    additional_headers: Mapping[str, str] | None = None,
    # Static HTTP headers to send on every request to the server.

    **kwargs,
    # Forwarded to toolbox_adk.ToolboxToolset.
)
```

Raises `ImportError` if `toolbox-adk` is not installed.

### Example 1 — load all tools from a local server

```python
import asyncio
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.toolbox_toolset import ToolboxToolset


async def main():
    toolset = ToolboxToolset("http://127.0.0.1:5000")

    agent = LlmAgent(
        name="toolbox_agent",
        model="gemini-2.0-flash",
        instruction="Use the available tools to complete the user's request.",
        tools=[toolset],
    )

    # Always close the toolset when done to release connections.
    await toolset.close()


asyncio.run(main())
```

### Example 2 — load a named toolset

```python
from google.adk.tools.toolbox_toolset import ToolboxToolset
from google.adk.agents.llm_agent import LlmAgent

# Only load the "database" toolset from the server (subset of all tools)
db_toolset = ToolboxToolset(
    "http://toolbox-server:5000",
    toolset_name="database",
)

agent = LlmAgent(
    name="db_agent",
    model="gemini-2.0-flash",
    instruction="Answer database queries using the provided tools.",
    tools=[db_toolset],
)
```

### Example 3 — select specific tools by name

```python
from google.adk.tools.toolbox_toolset import ToolboxToolset
from google.adk.agents.llm_agent import LlmAgent

# Only expose search_products and get_inventory — not all server tools
scoped_toolset = ToolboxToolset(
    "http://toolbox-server:5000",
    tool_names=["search_products", "get_inventory"],
)

agent = LlmAgent(
    name="catalogue_agent",
    model="gemini-2.0-flash",
    instruction="Help users find products and check stock levels.",
    tools=[scoped_toolset],
)
```

### Example 4 — auth token getter for per-request JWT injection

```python
import os
import asyncio
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.toolbox_toolset import ToolboxToolset
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService


def get_api_token() -> str:
    """Return a fresh API token (called per-request by the toolset)."""
    return os.environ.get("TOOLBOX_API_TOKEN", "")


async def main():
    toolset = ToolboxToolset(
        "http://toolbox-server:5000",
        auth_token_getters={"my-auth-service": get_api_token},
    )

    agent = LlmAgent(
        name="authed_agent",
        model="gemini-2.0-flash",
        instruction="Use the tools to complete requests.",
        tools=[toolset],
    )

    try:
        runner = Runner(
            agent=agent,
            app_name="authed_app",
            session_service=InMemorySessionService(),
        )
        print("Runner ready with authenticated toolset.")
    finally:
        await toolset.close()


asyncio.run(main())
```

### Example 5 — bound params to fix a shared parameter across all tools

```python
from google.adk.tools.toolbox_toolset import ToolboxToolset
from google.adk.agents.llm_agent import LlmAgent
import os


def get_tenant_id() -> str:
    """Return the current tenant ID from environment."""
    return os.environ.get("TENANT_ID", "default")


# All tools on the server that accept a "tenant_id" parameter
# will have it pre-bound — the LLM never needs to supply it.
tenanted_toolset = ToolboxToolset(
    "http://toolbox-server:5000",
    bound_params={
        "tenant_id": get_tenant_id,         # dynamic callable
        "api_version": "v2",                # static value
    },
)

agent = LlmAgent(
    name="tenanted_agent",
    model="gemini-2.0-flash",
    instruction="Process requests for the current tenant.",
    tools=[tenanted_toolset],
)
```

---

## 7 · `JoinNode` — workflow fan-in synchronization

**Source:** `google.adk.workflow._join_node`

`JoinNode` is a `BaseNode` subclass with a single property override: `_requires_all_predecessors` returns `True`. The workflow orchestrator uses this flag to withhold dispatching the node until **every** predecessor branch has completed and delivered its output. When all predecessors have reported in, the orchestrator calls `_run_impl` with `node_input` as a dict keyed by predecessor node name. `JoinNode._run_impl` immediately re-emits that dict as its own output — it is a transparent aggregator, not a transformer. The collected dict is then available to the next nodes in the graph.

`JoinNode` also overrides `_validate_input_data` to apply `input_schema` validation to each value in the dict independently (rather than to the dict as a whole).

### Constructor (source-verified)

```python
from google.adk.workflow._join_node import JoinNode

JoinNode(
    name: str,
    # Inherits all BaseNode fields: input_schema, output_schema,
    # retry_config, timeout, rerun_on_resume, etc.
)
```

### Example 1 — basic fan-out / fan-in pattern

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow._join_node import JoinNode
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START


async def summarise(ctx, node_input):
    """Merge outputs from parallel branches."""
    # node_input is {"translate_fr": "...", "translate_de": "..."}
    combined = "\n".join(f"{k}: {v}" for k, v in node_input.items())
    yield combined


translate_fr = LlmAgent(
    name="translate_fr",
    model="gemini-2.0-flash",
    instruction="Translate the input text to French.",
)
translate_de = LlmAgent(
    name="translate_de",
    model="gemini-2.0-flash",
    instruction="Translate the input text to German.",
)

join = JoinNode(name="join_translations")

wf = Workflow(
    name="parallel_translate",
    edges=[
        (START, (translate_fr, translate_de)),  # tuple fan-out: NodeLike objects
        (translate_fr, join),
        (translate_de, join),                   # fan-in
        (join, summarise),
    ],
)
```

### Example 2 — `JoinNode` output dict structure

```python
from google.adk.workflow._join_node import JoinNode
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START


async def branch_a(ctx, node_input):
    yield {"score": 0.9, "label": "positive"}


async def branch_b(ctx, node_input):
    yield {"score": 0.3, "label": "negative"}


async def aggregate(ctx, node_input):
    # node_input == {"branch_a": {"score": 0.9, "label": "positive"},
    #                "branch_b": {"score": 0.3, "label": "negative"}}
    scores = [v["score"] for v in node_input.values()]
    avg = sum(scores) / len(scores)
    yield {"average_score": avg}


join = JoinNode(name="join")

wf = Workflow(
    name="score_aggregator",
    edges=[
        (START, (branch_a, branch_b)),  # tuple fan-out: NodeLike objects
        (branch_a, join),
        (branch_b, join),
        (join, aggregate),
    ],
)
```

### Example 3 — three-branch fan-in with schema validation

```python
from pydantic import BaseModel
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow._join_node import JoinNode
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START


class SearchResult(BaseModel):
    query: str
    hits: int
    top_result: str


search_web = LlmAgent(
    name="search_web",
    model="gemini-2.0-flash",
    instruction="Search the web and return top result.",
    output_schema=SearchResult,
)
search_wiki = LlmAgent(
    name="search_wiki",
    model="gemini-2.0-flash",
    instruction="Search Wikipedia and return top result.",
    output_schema=SearchResult,
)
search_news = LlmAgent(
    name="search_news",
    model="gemini-2.0-flash",
    instruction="Search recent news and return top result.",
    output_schema=SearchResult,
)

# JoinNode collects all three SearchResult dicts before proceeding
join = JoinNode(name="join_search")


async def synthesise(ctx, node_input):
    # node_input: {"search_web": {...}, "search_wiki": {...}, "search_news": {...}}
    total_hits = sum(v["hits"] for v in node_input.values())
    yield {"total_hits": total_hits, "sources": list(node_input.keys())}


wf = Workflow(
    name="multi_search",
    edges=[
        (START, (search_web, search_wiki, search_news)),  # tuple fan-out
        (search_web, join),
        (search_wiki, join),
        (search_news, join),
        (join, synthesise),
    ],
)
```

---

## 8 · `@node` decorator + `FunctionNode` — workflow node creation

**Source:** `google.adk.workflow._node`

The `node` function is both a decorator and a factory. Used as `@node` it wraps a plain async generator function in a `FunctionNode`. Used as `@node(name=..., retry_config=..., parallel_worker=True)` it is a decorator factory. Used as `node(existing_agent_or_tool)` it calls `workflow_graph_utils.build_node()` to wrap any `NodeLike` (a `BaseAgent`, `BaseTool`, `BaseNode`, or async generator callable). When `parallel_worker=True`, the result is always a `_ParallelWorker` wrapping the built node.

The module also exposes `Node`, a `BaseNode` subclass designed for class-based workflow nodes — override `run_node_impl` and set `parallel_worker=True` as a class field to get fan-out behaviour without the decorator.

### Signature (source-verified)

```python
from google.adk.workflow._node import node, Node

# As a decorator:
@node
async def my_func(ctx, node_input): ...

# As a decorator factory:
@node(name="custom_name", retry_config=RetryConfig(...), timeout=30.0)
async def my_func(ctx, node_input): ...

# With parallel_worker:
@node(parallel_worker=True)
async def process_item(ctx, node_input): ...

# As a factory on an existing NodeLike:
wrapped = node(my_llm_agent, name="renamed_agent")
parallel_agent = node(my_llm_agent, parallel_worker=True)

# Class-based:
class MyNode(Node):
    parallel_worker: bool = True
    async def run_node_impl(self, *, ctx, node_input):
        yield node_input  # override to implement node logic
```

`auth_config` requires `rerun_on_resume=True` (auth flows may interrupt and resume).

### Example 1 — simple `@node` decorator

```python
from google.adk.workflow._node import node
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START


@node
async def fetch_data(ctx, node_input):
    """Simulate fetching data from an API."""
    # node_input is whatever the preceding node yielded
    topic = node_input if isinstance(node_input, str) else "default"
    yield {"topic": topic, "records": [1, 2, 3, 4, 5]}


@node
async def process_data(ctx, node_input):
    """Process the fetched records."""
    records = node_input.get("records", [])
    yield {"count": len(records), "total": sum(records)}


wf = Workflow(
    name="data_pipeline",
    edges=[(START, fetch_data), (fetch_data, process_data)],
)
```

### Example 2 — `@node()` factory with retry and timeout

```python
from google.adk.workflow._node import node
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START


@node(
    name="flaky_api_call",
    retry_config=RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        backoff_multiplier=2.0,
        jitter=0.1,
    ),
    timeout=10.0,
)
async def call_external_api(ctx, node_input):
    """Call an external API with retry logic."""
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/data/{node_input}")
        response.raise_for_status()
        yield response.json()


wf = Workflow(
    name="resilient_pipeline",
    edges=[(START, call_external_api)],
)
```

### Example 3 — `node()` factory wrapping an existing `LlmAgent`

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow._node import node
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START

classifier = LlmAgent(
    name="classifier",
    model="gemini-2.0-flash",
    instruction="Classify the sentiment of the input text as positive, negative, or neutral.",
)

# Wrap with node() to override the name or add retry/timeout
# without modifying the original LlmAgent
wrapped_classifier = node(
    classifier,
    name="sentiment_step",
    timeout=15.0,
)

wf = Workflow(
    name="sentiment_pipeline",
    edges=[(START, wrapped_classifier)],
)
```

### Example 4 — `@node(rerun_on_resume=True, auth_config=...)` for OAuth nodes

```python
from google.adk.workflow._node import node
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START

GOOGLE_AUTH = AuthConfig(
    auth_scheme=OpenIdConnectWithConfig(
        authorization_endpoint="https://accounts.google.com/o/oauth2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/calendar.readonly"],
    ),
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id="YOUR_CLIENT_ID",
            client_secret="YOUR_CLIENT_SECRET",
        ),
    ),
)


@node(rerun_on_resume=True, auth_config=GOOGLE_AUTH)
async def fetch_calendar(ctx, node_input):
    """Fetch Google Calendar events after OAuth2 is complete."""
    cred = ctx.get_auth_response(GOOGLE_AUTH)
    if cred is None:
        yield {"status": "auth_pending"}
        return
    # Use cred.oauth2.access_token to call the Calendar API
    yield {"events": [], "status": "fetched"}


wf = Workflow(
    name="calendar_workflow",
    edges=[(START, fetch_calendar)],
)
```

### Example 5 — class-based `Node` with `parallel_worker=True`

```python
from google.adk.workflow._node import Node
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START


class EnrichRecord(Node):
    """Enrich a single record with metadata — runs in parallel for each list item."""

    parallel_worker: bool = True  # framework wraps this in _ParallelWorker

    async def run_node_impl(self, *, ctx, node_input):
        # node_input is one record dict when called by _ParallelWorker
        enriched = {**node_input, "enriched": True, "score": len(str(node_input))}
        yield enriched


enrich_node = EnrichRecord(name="enrich_records")


async def produce_records(ctx, node_input):
    """Produce a list of records for parallel enrichment."""
    yield [{"id": i, "value": i * 10} for i in range(5)]


wf = Workflow(
    name="batch_enrich",
    edges=[(START, produce_records), (produce_records, enrich_node)],
)
```

---

## 9 · `_ParallelWorker` — fan-out map over list inputs

**Source:** `google.adk.workflow._parallel_worker`

`_ParallelWorker` is a `BaseNode` subclass that maps a **single wrapped node** over every item in a list input. When `node_input` arrives it normalises it to a list (wrapping single items). It then dispatches `ctx.run_node(self._node, node_input=item, use_sub_branch=True)` as an `asyncio.Task` for each item, respecting `max_concurrency` via a semaphore-like loop that drains tasks as they complete and backfills from the remaining queue. Results are collected into a list that preserves input order. If any task raises an exception, all remaining pending tasks are cancelled before re-raising.

`_ParallelWorker` always sets `rerun_on_resume=True` on itself (because it spawns sub-branches that may individually pause and resume). It cannot wrap `START`.

### Constructor (source-verified)

```python
from google.adk.workflow._parallel_worker import _ParallelWorker

_ParallelWorker(
    *,
    node: NodeLike,              # the node to run once per list item
    max_concurrency: int | None = None,  # None = unlimited
    retry_config: RetryConfig | None = None,
    timeout: float | None = None,
)
```

In practice you do not construct `_ParallelWorker` directly — use `@node(parallel_worker=True)` or `node(existing, parallel_worker=True)` and the decorator handles construction.

### Example 1 — `@node(parallel_worker=True)` fan-out

```python
from google.adk.workflow._node import node
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START


@node(parallel_worker=True)
async def summarise_document(ctx, node_input):
    """Summarise one document. Runs in parallel for each item in the list."""
    # node_input is a single document dict
    text = node_input.get("text", "")
    yield {"doc_id": node_input.get("id"), "summary": text[:100] + "…"}


async def produce_documents(ctx, node_input):
    """Produce a list of documents for parallel summarisation."""
    yield [
        {"id": "doc1", "text": "Long document one " * 20},
        {"id": "doc2", "text": "Long document two " * 20},
        {"id": "doc3", "text": "Long document three " * 20},
    ]


wf = Workflow(
    name="batch_summariser",
    edges=[
        (START, produce_documents),
        (produce_documents, summarise_document),
    ],
)
```

### Example 2 — `max_concurrency` to throttle parallel tasks

```python
from google.adk.workflow._parallel_worker import _ParallelWorker
from google.adk.workflow._node import node
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START


@node
async def call_rate_limited_api(ctx, node_input):
    """Call an API that allows at most 3 concurrent requests."""
    import asyncio
    await asyncio.sleep(0.1)  # simulate network round-trip
    yield {"input": node_input, "result": f"processed_{node_input}"}


# Wrap in _ParallelWorker with max_concurrency=3
throttled_worker = _ParallelWorker(
    node=call_rate_limited_api,
    max_concurrency=3,
)


async def produce_ids(ctx, node_input):
    yield list(range(10))  # 10 items, max 3 concurrent


wf = Workflow(
    name="throttled_pipeline",
    edges=[
        (START, produce_ids),
        (produce_ids, throttled_worker),
    ],
)
```

### Example 3 — parallel `LlmAgent` calls via `node(agent, parallel_worker=True)`

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow._node import node
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START

translate_agent = LlmAgent(
    name="translate",
    model="gemini-2.0-flash",
    instruction=(
        "Translate the text in node_input['text'] to the language "
        "specified in node_input['target_language']."
    ),
)

# Run translate_agent once per item — all items run concurrently
parallel_translate = node(translate_agent, parallel_worker=True)


async def prepare_translations(ctx, node_input):
    text = node_input or "Hello, world!"
    yield [
        {"text": text, "target_language": "French"},
        {"text": text, "target_language": "German"},
        {"text": text, "target_language": "Japanese"},
    ]


wf = Workflow(
    name="multi_lang_translate",
    edges=[
        (START, prepare_translations),
        (prepare_translations, parallel_translate),
    ],
)
```

### Example 4 — single-item input auto-wrapping

```python
from google.adk.workflow._node import node
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START


@node(parallel_worker=True)
async def process_item(ctx, node_input):
    """Process one item — also handles single (non-list) input."""
    yield f"done:{node_input}"


async def send_single(ctx, node_input):
    # Not a list — _ParallelWorker wraps it as [node_input]
    yield "single_value"


wf = Workflow(
    name="single_item_workflow",
    edges=[
        (START, send_single),
        (send_single, process_item),
    ],
)
# Output from process_item: ["done:single_value"]  (always a list)
```

### Example 5 — error propagation: one failure cancels siblings

```python
import asyncio
from google.adk.workflow._node import node
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START


@node(parallel_worker=True)
async def unreliable_step(ctx, node_input):
    """Raises for item index 2; all other tasks are cancelled."""
    if node_input == 2:
        raise ValueError(f"Item {node_input} failed intentionally.")
    await asyncio.sleep(0.05)
    yield f"ok:{node_input}"


async def produce_items(ctx, node_input):
    yield list(range(5))


wf = Workflow(
    name="error_propagation_demo",
    edges=[
        (START, produce_items),
        (produce_items, unreliable_step),
    ],
)

# When run: _ParallelWorker catches the ValueError from task index 2,
# cancels tasks for indices 3 and 4, then re-raises the ValueError.
# The workflow terminates with an error — partial results are NOT returned.
```

---

## 10 · `Workflow.max_concurrency` + complex edge patterns

**Source:** `google.adk.workflow._workflow`

`Workflow` is a `BaseNode` subclass whose `_run_impl` **is** the orchestration loop — it seeds triggers for `START` successors, then continuously schedules ready nodes via `NodeRunner`, handles completions, and finalises by collecting terminal outputs. The two ways to define a workflow graph are: the `edges` field (a list of `EdgeItem` tuples compiled into a `Graph` at `model_post_init`) and the `graph` field (a pre-built `Graph`). You cannot set both.

`max_concurrency` throttles only **static graph edges** — nodes scheduled by the orchestrator. Dynamic nodes dispatched via `ctx.run_node()` are always unlimited because they are awaited inline by their caller; throttling them would deadlock. `Workflow` always sets `rerun_on_resume=True` because the orchestration loop reconstructs static node states from session events on resume.

`mode='task'` `LlmAgent` instances may not appear as static graph nodes (the scheduler overwrites `node_input` on re-entry, losing the task brief). They must be used as chat sub-agents or dispatched dynamically.

### Model fields (source-verified)

```python
from google.adk.workflow._workflow import Workflow

class Workflow(BaseNode):
    rerun_on_resume: bool = True       # always True; set at class level

    edges: list[EdgeItem] = []
    # EdgeItem formats:
    #   (source, target)                              — simple edge
    #   (source, (target1, target2))                  — fan-out (tuple, not list)
    #   (source, {"value": node_callable, ...})       — RoutingMap conditional routing

    max_concurrency: int | None = None
    # Limits parallel static-graph nodes. None = unlimited.
    # Does NOT affect ctx.run_node() — those are always unlimited.

    graph: Graph | None = None
    # Provide a pre-compiled Graph instead of edges (mutually exclusive).
```

### Example 1 — `max_concurrency` to limit parallel static nodes

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START


async def split(ctx, node_input):
    """Fan out: send the same input to five parallel workers."""
    yield node_input


worker_nodes = [
    LlmAgent(name=f"worker_{i}", model="gemini-2.0-flash",
             instruction=f"Process task variant {i}.")
    for i in range(5)
]

# max_concurrency=2: only 2 of the 5 workers run at the same time.
# Fan-out uses a tuple of NodeLike objects, not a list of name strings.
wf = Workflow(
    name="bounded_fan_out",
    max_concurrency=2,
    edges=[
        (START, split),
        (split, tuple(worker_nodes)),   # tuple fan-out to all workers
    ],
)
```

### Example 2 — conditional routing with a RoutingMap

```python
from google.adk.events.event import Event
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START


async def classify_ticket(ctx, node_input):
    """Emit a route for RoutingMap dispatch, preserving the original payload.

    RoutingMap edges match against Event.actions.route, not Event.output.
    Include output=node_input so the selected handler receives the ticket text;
    without it child_ctx.output is None and handlers receive node_input=None.
    """
    text = str(node_input).lower()
    if "invoice" in text or "payment" in text:
        yield Event(output=node_input, route="billing")
    elif "error" in text or "crash" in text:
        yield Event(output=node_input, route="tech")
    else:
        yield Event(output=node_input, route="general")


async def handle_billing(ctx, node_input):
    yield f"Billing handled: {node_input}"


async def handle_tech(ctx, node_input):
    yield f"Tech handled: {node_input}"


async def handle_general(ctx, node_input):
    yield f"General handled: {node_input}"


wf = Workflow(
    name="ticket_router",
    edges=[
        (START, classify_ticket),
        (classify_ticket, {
            "billing": handle_billing,
            "tech": handle_tech,
            "general": handle_general,
        }),
    ],
)
```

### Example 3 — dict-based routing

```python
from google.adk.events.event import Event
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START


async def get_priority(ctx, node_input):
    """Emit a route and preserve payload so handlers receive the original input."""
    score = len(str(node_input))
    if score > 100:
        yield Event(output=node_input, route="high")
    elif score > 50:
        yield Event(output=node_input, route="medium")
    else:
        yield Event(output=node_input, route="low")


async def urgent_handler(ctx, node_input):
    yield f"URGENT: {node_input}"


async def standard_handler(ctx, node_input):
    yield f"standard: {node_input}"


async def low_handler(ctx, node_input):
    yield f"low priority: {node_input}"


wf = Workflow(
    name="priority_router",
    edges=[
        (START, get_priority),
        # RoutingMap: output value → target callable (NodeLike), not a string name
        (get_priority, {
            "high": urgent_handler,
            "medium": standard_handler,
            "low": low_handler,
        }),
    ],
)
```

### Example 4 — pre-compiled `Graph` object

```python
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._graph import Graph
from google.adk.workflow._base_node import START
from google.adk.agents.llm_agent import LlmAgent


extract = LlmAgent(name="extract", model="gemini-2.0-flash",
                   instruction="Extract named entities from the text.")
classify = LlmAgent(name="classify", model="gemini-2.0-flash",
                    instruction="Classify the extracted entities by type.")

# Graph has no add_edge(); use from_edge_items() for programmatic construction.
graph = Graph.from_edge_items([
    (START, extract),
    (extract, classify),
])
graph.validate_graph()   # raises if graph is invalid (cycles, orphans, etc.)

# Pass the pre-compiled graph — do NOT also set edges
wf = Workflow(
    name="nlp_pipeline",
    graph=graph,
)
```

### Example 5 — full pattern: fan-out → parallel work → fan-in → reduce

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow._node import node
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._base_node import START


async def split_topics(ctx, node_input):
    """Produce three research topics for parallel investigation."""
    yield ["climate change", "renewable energy", "carbon capture"]


@node(parallel_worker=True)
async def research_topic(ctx, node_input):
    """Research one topic — runs concurrently for all three."""
    yield {"topic": node_input, "summary": f"Summary of {node_input}"}


@node
async def collect_all(ctx, node_input):
    """Collect outputs from research_topic (list of dicts)."""
    yield node_input  # node_input is already a list from _ParallelWorker


async def write_report(ctx, node_input):
    """Combine all summaries into a final report."""
    summaries = [item["summary"] for item in node_input]
    yield "REPORT:\n" + "\n".join(f"- {s}" for s in summaries)


wf = Workflow(
    name="research_pipeline",
    max_concurrency=3,
    edges=[
        (START, split_topics),
        (split_topics, research_topic),   # _ParallelWorker maps over the list
        (research_topic, collect_all),
        (collect_all, write_report),
    ],
)
```

---

## Cross-reference

| Topic | Volume | Sections |
|---|---|---|
| `transfer_to_agent` free function, `FunctionTool` | [Vol. 1](./google_adk_class_deep_dives/) | `FunctionTool`, tool dispatch |
| `BaseTool`, `process_llm_request` protocol | [Vol. 3](./google_adk_class_deep_dives_v3/) | `BaseTool` internals |
| `BaseToolset`, `get_tools` protocol | [Vol. 5](./google_adk_class_deep_dives_v5/) | `BaseToolset` lifecycle |
| `LlmAgent.output_schema`, `output_key` | [Vol. 6](./google_adk_class_deep_dives_v6/) | `LlmAgent` structured output |
| Memory services: `InMemoryMemoryService`, `VertexAiMemoryBankService` | [Vol. 8](./google_adk_class_deep_dives_v8/) | Memory architecture |
| `LlamaIndexRetrieval` base class | [Vol. 9](./google_adk_class_deep_dives_v9/) | Retrieval tools |
| `MCPToolset`, `StdioConnectionParams`, `SseConnectionParams` | [Vol. 10](./google_adk_class_deep_dives_v10/) | MCP toolsets |
| `RetryConfig` (used by `@node`) | [Vol. 12](./google_adk_class_deep_dives_v12/) | Workflow retry |
| `BaseNode`, `START` sentinel | [Vol. 13](./google_adk_class_deep_dives_v13/) | Workflow nodes |
| `Graph`, `EdgeItem`, `validate_graph` | [Vol. 14](./google_adk_class_deep_dives_v14/) | Graph construction |
| `AuthConfig`, `auth_config` on nodes | [Vol. 15](./google_adk_class_deep_dives_v15/) | Node auth |
| `VertexAiRagRetrieval` (compare to `VertexAiSearchTool`) | [Vol. 17](./google_adk_class_deep_dives_v17/) | Grounding and retrieval |
| `Context` (unified `ToolContext`/`CallbackContext`) | [Vol. 18](./google_adk_class_deep_dives_v18/) | Context and HITL |
| `App`, `EventsCompactionConfig`, `ContextCacheConfig` | [Vol. 18](./google_adk_class_deep_dives_v18/) | App configuration |

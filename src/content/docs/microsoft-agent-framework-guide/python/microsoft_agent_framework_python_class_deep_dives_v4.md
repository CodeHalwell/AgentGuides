---
title: "Microsoft Agent Framework (Python) — 10-API Deep Dives Vol. 4 (1.18.0)"
description: "Source-verified deep dives for VectorStoreField, VectorStoreCollectionDefinition, InMemoryCollection, InMemoryStore, Filter, FilterGroup, SecretString, load_settings, create_agent_hooks_middleware, and GroupChatBuilder — all verified against agent-framework 1.18.0 source."
framework: microsoft-agent-framework
language: python
---

# agent-framework (Python) — 10-API Deep Dives Vol. 4

**Verified against:** `agent-framework==1.18.0`
**Python requirement:** 3.10+

This volume covers 10 additional public APIs (eight classes and two functions) spanning vector store modelling, portable filter expressions, settings management, AGENT-HOOKS-0.1 enforcement, and multi-agent group chat orchestration. Each section includes the full constructor or signature, every meaningful method, and self-contained runnable examples verified against the 1.18.0 source.

See [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) for `WorkflowViz`, `FileMemoryProvider`, `AgentModeProvider`, `BackgroundAgentsProvider`, `ToolApprovalMiddleware`, `SwitchCaseEdgeGroup`, `MessageInjectionMiddleware`, `ToolResultCompactionStrategy`, `SummarizationStrategy`, and `TokenBudgetComposedStrategy`.

See [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) for `FanInEdgeGroup`, `FanOutEdgeGroup`, `FunctionalWorkflow`, `FunctionalWorkflowAgent`, `FileCheckpointStorage`, `InMemoryCheckpointStorage`, `MCPStdioTool`, `MCPStreamableHTTPTool`, `SelectiveToolCallCompactionStrategy`, and `TodoProvider`.

See [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) for `WorkflowBuilder`, `SlidingWindowStrategy`, `TruncationStrategy`, `ContextWindowCompactionStrategy`, `LocalEvaluator`, `InlineSkill`, `FileAccessProvider`, `MemoryContextProvider`, `FileHistoryProvider`, and `MCPWebsocketTool`.

---

## 1. `VectorStoreField`

**Module:** `agent_framework._vectors` (re-exported via `agent_framework`)

`VectorStoreField` annotates one field in a vector-store model. Every field has a `field_type`: `"key"` (the record identifier), `"vector"` (a dense or binary embedding), or `"data"` (any scalar payload). Apply it as an `Annotated` metadata object on a dataclass, Pydantic model, or plain class decorated with `@vectorstoremodel`.

### Constructor (overloaded)

```python
# Key field
VectorStoreField(
    "key",
    *,
    name: str | None = None,
    type_: str | None = None,
    storage_name: str | None = None,
    is_auto_generated: bool = False,
    provider_annotations: Mapping[str, Any] | None = None,
)

# Data field
VectorStoreField(
    "data",              # default — may be omitted
    *,
    name: str | None = None,
    type_: str | None = None,
    storage_name: str | None = None,
    is_indexed: bool | None = None,
    is_full_text_indexed: bool | None = None,
    provider_annotations: Mapping[str, Any] | None = None,
)

# Vector field
VectorStoreField(
    "vector",
    *,
    name: str | None = None,
    dimensions: int,          # required
    index_kind: IndexKind | None = None,
    distance_function: DistanceFunction | None = None,
    embedding_generator: EmbeddingClient | None = None,
    storage_name: str | None = None,
    type_: str | None = None,
    provider_annotations: Mapping[str, Any] | None = None,
)
```

| Parameter | Notes |
|---|---|
| `field_type` | `"key"` / `"data"` / `"vector"`. Default `"data"`. |
| `name` | Field name in the model. Filled in by `@vectorstoremodel` when omitted. |
| `dimensions` | Required for vector fields. Positive integer giving the expected embedding length. |
| `index_kind` | `"hnsw"`, `"flat"`, `"ivf_flat"`, `"disk_ann"`, `"quantized_flat"`, `"dynamic"`, or `"default"`. |
| `distance_function` | `"cosine_similarity"`, `"cosine_distance"`, `"dot_prod"`, `"euclidean_distance"`, etc. |
| `embedding_generator` | Per-field embedding client. Overrides the collection-level generator. |
| `is_auto_generated` | Key fields only. The backing store generates the key when missing. |
| `storage_name` | Name used by the backing store when different from `name`. |
| `is_indexed` | Data fields. Whether the connector creates an index. |
| `is_full_text_indexed` | Data fields. Whether the connector creates a full-text index. |
| `provider_annotations` | Open dict for connector-specific options (e.g. `{"ef_search": 512}`). Copied on construction. |

### Attributes (all read-only after construction)

`field_type`, `name`, `type_`, `storage_name`, `is_indexed`, `is_full_text_indexed`, `dimensions`, `index_kind`, `distance_function`, `embedding_generator`, `is_auto_generated`, `provider_annotations`.

### Example — annotating a dataclass model

```python
from dataclasses import dataclass
from typing import Annotated
from agent_framework import VectorStoreField, vectorstoremodel

@vectorstoremodel(collection_name="articles")
@dataclass
class Article:
    id: Annotated[str, VectorStoreField("key")]
    title: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    body: Annotated[str, VectorStoreField("data")]
    embedding: Annotated[list[float], VectorStoreField(
        "vector",
        dimensions=1536,
        distance_function="cosine_similarity",
        index_kind="hnsw",
    )]

# @vectorstoremodel attaches the derived definition as a class attribute
defn = Article.__vectorstoremodel_definition__
print(defn.key_name)             # "id"
print(defn.vector_field_names)   # ["embedding"]
print(defn.data_field_names)     # ["title", "body"]
```

### Example — provider-specific annotations

```python
from dataclasses import dataclass
from typing import Annotated
from agent_framework import VectorStoreField, vectorstoremodel

@vectorstoremodel
@dataclass
class Chunk:
    chunk_id: Annotated[str, VectorStoreField("key")]
    content: Annotated[str, VectorStoreField("data")]
    vec: Annotated[list[float], VectorStoreField(
        "vector",
        dimensions=3072,
        distance_function="cosine_distance",
        provider_annotations={"m": 4, "ef_construction": 400},
    )]
```

---

## 2. `VectorStoreCollectionDefinition`

**Module:** `agent_framework._vectors` (re-exported via `agent_framework`)

`VectorStoreCollectionDefinition` carries the complete field roster for a collection. Most users obtain one through `@vectorstoremodel` or `register_vectorstoremodel`. Construct it directly for schema-less `dict` records.

### Constructor

```python
VectorStoreCollectionDefinition(
    fields: Sequence[VectorStoreField],
    *,
    collection_name: str | None = None,
)
```

Raises `ValueError` if: there are no fields, field names are empty or duplicated, storage names clash, or there is not exactly one `"key"` field.

### Properties

| Property | Returns | Notes |
|---|---|---|
| `names` | `list[str]` | All model field names. |
| `storage_names` | `list[str]` | Backing-store field names (falls back to `name`). |
| `key_name` | `str` | The single key field's model name. |
| `key_field` | `VectorStoreField` | The key field. |
| `key_field_storage_name` | `str` | The key field's storage name. |
| `vector_fields` | `list[VectorStoreField]` | All vector fields. |
| `data_fields` | `list[VectorStoreField]` | All data fields. |
| `vector_field_names` | `list[str]` | Names of vector fields. |
| `data_field_names` | `list[str]` | Names of data fields. |

### Methods

| Method | Returns | Notes |
|---|---|---|
| `get_names(*, include_vector_fields=True, include_key_field=True)` | `list[str]` | Filter the name list by field type. |
| `get_storage_names(*, include_vector_fields=True, include_key_field=True)` | `list[str]` | Same, but storage names. |
| `try_get_field(field_name)` | `VectorStoreField \| None` | Look up by model or storage name. |
| `try_get_vector_field(field_name=None)` | `VectorStoreField \| None` | First vector field if `field_name` is `None`. |

### Example — explicit definition for dict records

```python
from agent_framework import VectorStoreField, VectorStoreCollectionDefinition, InMemoryCollection

definition = VectorStoreCollectionDefinition(
    fields=[
        VectorStoreField("key", name="id"),
        VectorStoreField("data", name="text"),
        VectorStoreField("vector", name="embedding", dimensions=4),
    ],
    collection_name="notes",
)

# Drive an InMemoryCollection with raw dicts
collection: InMemoryCollection[str, dict] = InMemoryCollection(
    dict,
    definition=definition,
)

import asyncio

async def main():
    await collection.ensure_collection_exists()
    keys = await collection.upsert([
        {"id": "a", "text": "hello world", "embedding": [0.1, 0.2, 0.3, 0.4]},
    ], generate_vectors=False)  # embedding already supplied; no generator attached
    print(keys)  # ["a"]
    records = await collection.get(["a"])
    print(records[0]["text"])  # "hello world"

asyncio.run(main())
```

---

## 3. `InMemoryCollection`

**Module:** `agent_framework._in_memory` (re-exported via `agent_framework`)

`InMemoryCollection` is a dependency-free, process-local vector collection that implements both `BaseVectorCollection` (CRUD) and `BaseVectorSearch` (vector search). It stores records as deep copies of dictionaries, applies filters locally, and scores with one of eight distance functions. Intended for unit tests and quick-start prototyping — not thread-safe and not persistent.

### Constructor

```python
InMemoryCollection(
    record_type: type[ModelT],
    *,
    definition: VectorStoreCollectionDefinition | None = None,
    collection_name: str | None = None,
    embedding_generator: EmbeddingClient | None = None,
)
```

| Parameter | Notes |
|---|---|
| `record_type` | Application record type (dataclass, Pydantic model, or `dict`). |
| `definition` | Required when `record_type` is `dict` or lacks `@vectorstoremodel`. |
| `collection_name` | Overrides the model-derived name. |
| `embedding_generator` | Generates vectors for `upsert` and `search`. |

### Key methods (inherited from `BaseVectorCollection` + `BaseVectorSearch`)

| Method | Notes |
|---|---|
| `await ensure_collection_exists()` | Marks the collection as existing. No-op if already exists. |
| `await collection_exists()` | Returns `True` after `ensure_collection_exists()`. |
| `await ensure_collection_deleted()` | Clears all records and marks collection absent. |
| `await upsert(records, *, generate_vectors=True)` | Insert or overwrite a batch; returns list of keys. |
| `await get(keys=None, *, filter=None, top=10, skip=0, order_by=None, include_vectors=False)` | Retrieve by keys or list a filtered/sorted page. |
| `await delete(keys)` | Delete records by key. |
| `await search(values=None, *, vector=None, top=3, filter=None, score_threshold=None, ...)` | Vector search. Pass pre-computed `vector` or let the `embedding_generator` build it from `values`. Returns `SearchResults`. |

> **`create_vector_search_tool`** is a module-level function (imported from `agent_framework`), not a method of `InMemoryCollection`. Pass the collection as its first positional argument: `create_vector_search_tool(col, description="...", top=5)`. See Example 3 below.

Supported distance functions: `cosine_similarity`, `cosine_distance` (default), `dot_prod`, `negative_dot_prod`, `euclidean_distance`, `euclidean_squared_distance`, `manhattan`, `hamming`.

### Example 1 — CRUD with a dataclass model

```python
import asyncio
from dataclasses import dataclass
from typing import Annotated
from agent_framework import VectorStoreField, vectorstoremodel, InMemoryCollection

@vectorstoremodel(collection_name="products")
@dataclass
class Product:
    sku: Annotated[str, VectorStoreField("key")]
    name: Annotated[str, VectorStoreField("data")]
    vec: Annotated[list[float], VectorStoreField("vector", dimensions=3)]

async def main():
    col: InMemoryCollection[str, Product] = InMemoryCollection(Product)
    await col.ensure_collection_exists()

    keys = await col.upsert([
        Product(sku="A1", name="Widget", vec=[0.1, 0.5, 0.3]),
        Product(sku="B2", name="Gadget", vec=[0.9, 0.1, 0.4]),
    ], generate_vectors=False)   # vectors already supplied
    print(keys)  # ["A1", "B2"]

    products = await col.get(["A1"])
    print(products[0].name)  # "Widget"

    await col.delete(["A1"])
    print(await col.get(["A1"]))  # []

asyncio.run(main())
```

### Example 2 — Vector search with `search()`

```python
import asyncio
from dataclasses import dataclass
from typing import Annotated
from agent_framework import VectorStoreField, vectorstoremodel, InMemoryCollection, SearchResults, SearchResponse

@vectorstoremodel
@dataclass
class Doc:
    id: Annotated[str, VectorStoreField("key")]
    text: Annotated[str, VectorStoreField("data")]
    embedding: Annotated[list[float], VectorStoreField("vector", dimensions=4, distance_function="cosine_similarity")]

async def main():
    col: InMemoryCollection[str, Doc] = InMemoryCollection(Doc)
    await col.ensure_collection_exists()
    await col.upsert([
        Doc(id="d1", text="cats", embedding=[1.0, 0.0, 0.0, 0.0]),
        Doc(id="d2", text="dogs", embedding=[0.0, 1.0, 0.0, 0.0]),
        Doc(id="d3", text="birds", embedding=[0.0, 0.0, 1.0, 0.0]),
    ], generate_vectors=False)

    results: SearchResults[SearchResponse[Doc]] = await col.search(
        vector=[1.0, 0.0, 0.0, 0.0],   # nearest to "cats"
        top=2,
    )
    async for result in results.results:  # results.results is AsyncIterable
        print(result.record.text, result.score)
    # cats  1.0  (exact match)
    # dogs  0.0  (orthogonal)

asyncio.run(main())
```

### Example 3 — Expose as an agent tool

```python
from agent_framework import Agent, InMemoryCollection, create_vector_search_tool
from agent_framework.openai import OpenAIChatClient

col: InMemoryCollection = ...  # already populated

search_tool = create_vector_search_tool(  # top-level function; col is the first positional arg
    col,
    description="Search product catalogue by semantic similarity",
    top=5,
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You can search the product catalogue.",
    tools=[search_tool],
)
```

---

## 4. `InMemoryStore`

**Module:** `agent_framework._in_memory` (re-exported via `agent_framework`)

`InMemoryStore` is the factory companion to `InMemoryCollection`. Multiple clients for the same collection name share the same in-memory state — useful when different parts of an application need independent `InMemoryCollection` handles that write to the same underlying dictionary.

### Constructor

```python
InMemoryStore(
    *,
    embedding_generator: EmbeddingClient | None = None,
)
```

### Methods

| Method | Returns | Notes |
|---|---|---|
| `get_collection(record_type, *, definition=None, collection_name=None, embedding_generator=None)` | `InMemoryCollection` | Creates or returns a client sharing state for `collection_name`. Raises `ValueError` if the name is already registered with a different definition. |
| `await list_collection_names()` | `Sequence[str]` | Names of all collections for which `ensure_collection_exists()` has been called. |
| `await collection_exists(name)` | `bool` | True if the named collection exists. |
| `await ensure_collection_deleted(name)` | — | Clears records and marks the collection absent. |

### Example — shared state across multiple clients

```python
import asyncio
from dataclasses import dataclass
from typing import Annotated
from agent_framework import VectorStoreField, vectorstoremodel, InMemoryStore

@vectorstoremodel(collection_name="kb")
@dataclass
class KBEntry:
    key: Annotated[str, VectorStoreField("key")]
    content: Annotated[str, VectorStoreField("data")]
    vec: Annotated[list[float], VectorStoreField("vector", dimensions=3)]

async def main():
    store = InMemoryStore()

    writer = store.get_collection(KBEntry)
    reader = store.get_collection(KBEntry)   # same underlying state

    await writer.ensure_collection_exists()
    await writer.upsert([KBEntry(key="k1", content="hello", vec=[1.0, 0.0, 0.0])],
                        generate_vectors=False)

    entries = await reader.get(["k1"])
    print(entries[0].content)   # "hello"

    names = await store.list_collection_names()
    print(names)  # ["kb"]

asyncio.run(main())
```

---

## 5. `Filter`

**Module:** `agent_framework._vector_filters` (re-exported via `agent_framework`)

`Filter` is the leaf node of a portable vector-store filter expression. It pairs a model field name with an operator and a value. Use it wherever a `FilterExpression` is accepted — `BaseVectorCollection.get()`, `BaseVectorSearch.search()` — and on the agent-facing `create_vector_search_tool()` with parametric `Param` values.

### Constructor

```python
Filter(
    field_name: str,
    operator: FilterOperator,
    value: Any = None,
)
```

`FilterOperator` is a `Literal` union of the following standard operators (provider-namespaced strings like `"azure.gte_score"` are also accepted):

| Category | Operators |
|---|---|
| Equality | `"eq"`, `"ne"` |
| Ordered | `"gt"`, `"gte"`, `"lt"`, `"lte"` |
| Range | `"between"` — value must be `(lower, upper)` |
| Set | `"in"`, `"not_in"` |
| Null / presence | `"is_null"`, `"is_not_null"`, `"exists"` |
| Collection | `"contains"`, `"contains_any"`, `"contains_all"` |
| Text | `"starts_with"`, `"ends_with"`, `"contains_text"` |

`is_null`, `is_not_null`, and `exists` accept no value — pass them without the third argument.

### Attributes

`field_name: str`, `operator: FilterOperator`, `value: Any`.

### Example — basic filters

```python
import asyncio
from dataclasses import dataclass
from typing import Annotated
from agent_framework import Filter, VectorStoreField, vectorstoremodel, InMemoryCollection

@vectorstoremodel
@dataclass
class Item:
    id: Annotated[str, VectorStoreField("key")]
    category: Annotated[str, VectorStoreField("data")]
    price: Annotated[float, VectorStoreField("data")]
    tags: Annotated[list[str], VectorStoreField("data")]
    vec: Annotated[list[float], VectorStoreField("vector", dimensions=2)]

async def main():
    col: InMemoryCollection[str, Item] = InMemoryCollection(Item)
    await col.ensure_collection_exists()
    await col.upsert([
        Item(id="i1", category="electronics", price=299.0, tags=["new", "sale"], vec=[1.0, 0.0]),
        Item(id="i2", category="books",       price=14.5,  tags=["classic"],     vec=[0.0, 1.0]),
        Item(id="i3", category="electronics", price=49.99, tags=["refurb"],      vec=[0.9, 0.1]),
    ], generate_vectors=False)

    # Price between 10 and 100
    cheap = await col.get(filter=Filter("price", "between", (10.0, 100.0)), top=10)
    print([x.id for x in cheap])  # ["i2", "i3"]

    # Category equals "electronics"
    elec = await col.get(filter=Filter("category", "eq", "electronics"), top=10)
    print([x.id for x in elec])   # ["i1", "i3"]

    # Tags contains "sale"
    sale = await col.get(filter=Filter("tags", "contains", "sale"), top=10)
    print([x.id for x in sale])   # ["i1"]

asyncio.run(main())
```

### Example — parametric filter for a search tool

```python
from agent_framework import Filter, Param, VectorStoreField, vectorstoremodel, InMemoryCollection, create_vector_search_tool

# Param("min_price", float | None, default=None, omit_if_none=True) means:
# the tool schema exposes "min_price" as an optional float parameter.
# When the model doesn't supply it, the filter leaf is omitted.
price_filter = Filter("price", "gte", Param("min_price", float | None, default=None, omit_if_none=True))

col = ...  # InMemoryCollection already set up
tool = create_vector_search_tool(  # top-level function; col is the first positional arg
    col,
    description="Search items; optionally filter by minimum price.",
    filter=price_filter,
    top=5,
)
```

---

## 6. `FilterGroup`

**Module:** `agent_framework._vector_filters` (re-exported via `agent_framework`)

`FilterGroup` composes one or more `Filter` / `FilterGroup` nodes with an explicit boolean operator: `"and"` (all must match), `"or"` (any must match), or `"not"` (negation of **exactly one** child — the `"not"` form intentionally accepts only a single child).

### Constructor

```python
FilterGroup(
    operator: FilterGroupOperator,   # "and" | "or" | "not"
    filters: Sequence[Filter | FilterGroup],
)
```

Raises `ValueError` if `"not"` receives anything other than one child, or if the `filters` sequence is empty or exceeds the structural limit.

### Attributes

`operator: FilterGroupOperator`, `filters: tuple[Filter | FilterGroup, ...]`.

### Example — AND / OR composition

```python
import asyncio
from dataclasses import dataclass
from typing import Annotated
from agent_framework import Filter, FilterGroup, VectorStoreField, vectorstoremodel, InMemoryCollection

@vectorstoremodel
@dataclass
class Product:
    id: Annotated[str, VectorStoreField("key")]
    category: Annotated[str, VectorStoreField("data")]
    price: Annotated[float, VectorStoreField("data")]
    in_stock: Annotated[bool, VectorStoreField("data")]
    vec: Annotated[list[float], VectorStoreField("vector", dimensions=2)]

async def main():
    col: InMemoryCollection[str, Product] = InMemoryCollection(Product)
    await col.ensure_collection_exists()
    await col.upsert([
        Product("p1", "electronics", 499.0, True,  [1.0, 0.0]),
        Product("p2", "electronics", 29.99, False, [0.9, 0.1]),
        Product("p3", "books",        9.99, True,  [0.0, 1.0]),
    ], generate_vectors=False)

    # (category == "electronics") AND (in_stock == True)
    in_stock_electronics = FilterGroup("and", [
        Filter("category", "eq", "electronics"),
        Filter("in_stock",  "eq", True),
    ])
    results = await col.get(filter=in_stock_electronics, top=10)
    print([r.id for r in results])  # ["p1"]

    # price < 15 OR category == "electronics"
    broad = FilterGroup("or", [
        Filter("price",    "lt", 15.0),
        Filter("category", "eq", "electronics"),
    ])
    results = await col.get(filter=broad, top=10)
    print([r.id for r in results])  # ["p1", "p2", "p3"]

asyncio.run(main())
```

### Example — NOT negation

```python
from agent_framework import Filter, FilterGroup

# Anything that is NOT out-of-stock
in_stock = FilterGroup("not", [Filter("in_stock", "eq", False)])
```

### Example — nested AND inside OR

```python
from agent_framework import Filter, FilterGroup

# (category == "electronics" AND price < 50) OR category == "books"
compound = FilterGroup("or", [
    FilterGroup("and", [
        Filter("category", "eq", "electronics"),
        Filter("price",    "lt", 50.0),
    ]),
    Filter("category", "eq", "books"),
])
```

---

## 7. `SecretString`

**Module:** `agent_framework._settings` (re-exported via `agent_framework`)

`SecretString` wraps a credential so that `str()`, `repr()`, f-string formatting, and string concatenation all emit `'**********'` instead of the real value. Use `get_secret_value()` to extract it for SDK calls.

### Constructor

```python
SecretString(value: str | SecretString)
```

Raises `TypeError` if `value` is neither a `str` nor a `SecretString`. Immutable after construction — `__setattr__` raises `AttributeError`.

### Methods and dunder behaviour

| Expression | Result |
|---|---|
| `str(s)` | `"**********"` |
| `repr(s)` | `"SecretString('**********')"` |
| `f"key={s}"` | `"key=**********"` |
| `"Bearer " + s` | `"Bearer **********"` |
| `s.get_secret_value()` | The raw string — use only when passing to an SDK. |
| `len(s)` | Length of the underlying value. |
| `s == other` | Compares underlying values. Accepts `str` or `SecretString`. |
| `hash(s)` | Hash of the underlying value. |
| `copy.copy(s)` | Returns `s` unchanged (immutable). |
| `copy.deepcopy(s)` | Returns `s` unchanged (immutable). |

### Example

```python
from agent_framework import SecretString

api_key = SecretString("sk-super-secret-1234")

# Safe in logs, print, repr:
print(api_key)           # **********
print(repr(api_key))     # SecretString('**********')
print(f"key={api_key}")  # key=**********

# Safe in concatenation — masking propagates:
header = "Authorization: Bearer " + api_key
print(header)            # Authorization: Bearer **********

# Length check without exposing value:
print(len(api_key) > 10)  # True

# Only expose when actually needed:
real_key = api_key.get_secret_value()   # "sk-super-secret-1234"
```

### Example — in a settings TypedDict

```python
from agent_framework import SecretString, load_settings
from typing import TypedDict

class MySettings(TypedDict, total=False):
    api_key: SecretString | None
    model: str | None

settings = load_settings(
    MySettings,
    env_prefix="MY_APP_",
    required_fields=["model"],
    model="gpt-4o",
    api_key="sk-test-key",   # explicit override — wrapped as SecretString automatically
)
# settings["api_key"] is a SecretString — safe to log, safe to store in memory
# Without the override, api_key would be None (no MY_APP_API_KEY env var set)
```

---

## 8. `load_settings`

**Module:** `agent_framework._settings` (re-exported via `agent_framework`)

`load_settings` builds a `TypedDict` instance from environment variables, a `.env` file, and explicit overrides. It replaces the older `AFBaseSettings` Pydantic-settings class with a zero-Pydantic-settings-dependency approach.

### Signature

```python
load_settings(
    settings_type: type[SettingsT],
    *,
    env_prefix: str = "",
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
    required_fields: Sequence[str | tuple[str, ...]] | None = None,
    **overrides: Any,
) -> SettingsT
```

| Parameter | Notes |
|---|---|
| `settings_type` | A `TypedDict` class whose keys map to setting names. |
| `env_prefix` | Prefix prepended to every key when looking up env vars (e.g. `"MY_APP_"` → `MY_APP_API_KEY`). |
| `env_file_path` | Path to a `.env` file. `None` (default) disables dotenv loading. |
| `env_file_encoding` | Encoding for the `.env` file. `None` uses the system default. |
| `required_fields` | List of required field names (strings) or mutual-exclusion groups (`tuples`). A tuple means "exactly one of these must be set". Raises `SettingNotFoundError` on failure. |
| `**overrides` | Explicit values that override env vars and dotenv. Validated against the TypedDict type. |

**Resolution order (highest wins):** overrides → `env_file_path` dotenv file (when specified) → environment variables → `None` for optional fields. The dotenv file takes precedence over process environment variables — use explicit overrides to win over both.

Fields typed `SecretString` are automatically wrapped. Fields typed `int`, `float`, or `bool` are coerced from their string env-var form.

### Example — full settings load

```python
import os
from typing import TypedDict
from agent_framework import SecretString, load_settings

class AppSettings(TypedDict, total=False):
    openai_api_key: SecretString | None
    azure_endpoint: str | None
    model: str | None
    max_tokens: int | None

# Simulate env vars:
os.environ["APP_MODEL"] = "gpt-4o-mini"
os.environ["APP_MAX_TOKENS"] = "512"
os.environ["APP_OPENAI_API_KEY"] = "sk-abc123"

settings = load_settings(
    AppSettings,
    env_prefix="APP_",
    required_fields=["model", ("openai_api_key", "azure_endpoint")],
    # override example — replaces env var:
    max_tokens=1024,
)

print(settings["model"])          # "gpt-4o-mini"
print(settings["max_tokens"])     # 1024  (override wins)
print(settings["openai_api_key"]) # **********  (SecretString masked)
print(settings["openai_api_key"].get_secret_value())  # "sk-abc123"
```

### Example — mutual-exclusion requirement

```python
from typing import TypedDict
from agent_framework import SecretString, load_settings

class SourceSettings(TypedDict, total=False):
    source_a: str | None
    source_b: str | None
    model: str | None

# Exactly one of "source_a" or "source_b" must be set — supply one via override:
settings = load_settings(
    SourceSettings,
    env_prefix="MY_",
    required_fields=[("source_a", "source_b")],
    source_a="db://localhost/main",   # satisfies the mutual-exclusion constraint
)
# Providing both raises SettingNotFoundError("mutually exclusive"):
# load_settings(SourceSettings, required_fields=[("source_a", "source_b")],
#               source_a="x", source_b="y")  # → SettingNotFoundError
```

---

## 9. `create_agent_hooks_middleware`

**Module:** `agent_framework._agent_hooks` (re-exported via `agent_framework`)

`create_agent_hooks_middleware` wires the [AGENT-HOOKS-0.1 protocol](https://github.com/responsibleai/agent-hooks) into an agent as a `MiddlewareBundle`. The bundle spans **three** middleware layers (agent, chat, function) and keeps them indivisible — installing part of the bundle would enforce only part of the control contract.

> **Trust-model caveat:** AGENT-HOOKS-0.1 is a cooperative in-process control contract, not a security boundary. The eight interception points provide best-effort mediation; complete mediation is not guaranteed by the spec. Do not rely solely on this middleware to enforce hard security invariants across trust boundaries.

Requires the optional `agent-hooks-sdk` package: `pip install agent-hooks-sdk`.

### Signature

```python
from agent_framework import create_agent_hooks_middleware
from agent_framework._types import MiddlewareBundle   # return type

create_agent_hooks_middleware(
    interceptors: Sequence[Interceptor] | Mapping[str, Interceptor],
    *,
    resolver: ApprovalResolver | None = None,
    mode: "enforce" | "evaluate_only" | EnforcementMode = "enforce",
    composition: CompositionConfig | None = None,
    identity_provider: str | IdentityProvider | None = "jcs-sha256",
    timeout: float | None = 5.0,
    record_sink: Callable[[InterceptionRecord], None] | None = None,
) -> MiddlewareBundle
```

| Parameter | Notes |
|---|---|
| `interceptors` | One or more objects implementing the `Interceptor` protocol. Pass a `Mapping` to give each interceptor a human-readable name (appears in violation summaries). At least one is required. |
| `resolver` | Consult this `ApprovalResolver` for liftable denies (human-in-the-loop approval flows). |
| `mode` | `"enforce"` (default): honours all verdicts. `"evaluate_only"`: records verdicts but does not act on them — useful for dry-run / shadow mode. |
| `composition` | Composition profile controlling how multiple interceptors' verdicts are combined. `None` uses the SDK default (`sequential/first_deny`). |
| `identity_provider` | `"jcs-sha256"` (default) hashes content for identity-bound records. `None` for unbound. |
| `timeout` | Per-interceptor/resolver call timeout (seconds). Default 5.0. |
| `record_sink` | Callback receiving every `InterceptionRecord` — ideal for audit logging. |

**Installation:** pass the returned `MiddlewareBundle` as the **first** (outermost) element of `Agent(middleware=[bundle, ...])`. Install exactly one bundle per agent; stacked bundles are rejected fail-closed.

### Enforcement points

| Point | When fired |
|---|---|
| `agent_startup` | Before the agent processes any input. |
| `input` | After input messages are assembled. |
| `pre_model_call` | Immediately before the LLM call. |
| `post_model_call` | After the LLM response but before tool execution. |
| `pre_tool_call` | Before each tool function runs. |
| `post_tool_call` | After each tool result is produced. |
| `output` | Before the final response is returned to the caller — also gates streaming release. |
| `agent_shutdown` | After the run completes (including error cases). |

### Example — simple content guard

```python
import asyncio
from agent_framework import Agent, create_agent_hooks_middleware
from agent_framework.openai import OpenAIChatClient

# agent-hooks-sdk must be installed: pip install agent-hooks-sdk
try:
    from agent_hooks import ALLOW, Verdict
except ImportError:
    print("Install agent-hooks-sdk to run this example")
    raise

class EgressGuard:
    """Block output (only) that contains the word 'secret'."""
    def intercept(self, context):
        # Guard only the output point — other points see instructions/tool data, not final response
        if context.get("interception_point") != "output":
            return ALLOW
        target = str(context.get("target", ""))
        if "secret" in target.lower():
            return Verdict.deny(reason="egress_blocked: sensitive content detected")
        return ALLOW

records = []

bundle = create_agent_hooks_middleware(
    {"egress_guard": EgressGuard()},
    record_sink=records.append,   # collect all interception records
    timeout=3.0,
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    middleware=[bundle],           # bundle goes first
)

async def main():
    response = await agent.run("Say the word 'secret'")
    # If the guard blocks, InterceptionBlocked is raised
    print(response.text)

asyncio.run(main())
```

### Example — evaluate_only (shadow / audit mode)

```python
from agent_framework import Agent, create_agent_hooks_middleware

audit_log = []

bundle = create_agent_hooks_middleware(
    [my_policy_interceptor],
    mode="evaluate_only",               # record verdicts but never block
    record_sink=audit_log.append,
)

agent = Agent(client=..., middleware=[bundle])
```

### Example — per-run session

`create_agent_hooks_middleware` creates a fresh `InterceptionEmitter` per `agent.run(...)`. To scope one session across multiple runs (stateful interceptors, shared approval ledger), use `create_agent_hooks_middleware_from_emitter` instead:

```python
from agent_framework import create_agent_hooks_middleware_from_emitter
from agent_hooks import InterceptionEmitter, AgentContextBuilder

emitter = InterceptionEmitter(interceptors=[my_interceptor])
builder = AgentContextBuilder(emitter=emitter)

bundle = create_agent_hooks_middleware_from_emitter(emitter, builder)
# Same bundle used for every run — one shared session
```

---

## 10. `GroupChatBuilder`

**Module:** `agent_framework_orchestrations` (install: `pip install agent-framework-orchestrations`)

**Import:** `from agent_framework_orchestrations import GroupChatBuilder`

`GroupChatBuilder` wires multiple agents (and custom `Executor` nodes) into a star-topology group chat where an orchestrator dynamically selects the next speaker each round. It mirrors the ergonomics of `SequentialBuilder` and `ConcurrentBuilder` from the same package.

Three orchestrator modes:
- `selection_func` — a simple Python callable receives `GroupChatState` and returns the next participant name.
- `orchestrator_agent` — an `Agent` instance picks the next speaker using structured output (requires structured-output support on the model).
- `orchestrator` — a custom `BaseGroupChatOrchestrator` subclass.

### Constructor

```python
GroupChatBuilder(
    *,
    participants: Sequence[SupportsAgentRun | Executor] | None = None,
    participant_factories: Sequence[Callable[[], SupportsAgentRun | Executor]] | None = None,
    # Orchestrator — exactly one required
    orchestrator_agent: Agent | Callable[[], Agent] | None = None,
    orchestrator: BaseGroupChatOrchestrator | Callable[[], BaseGroupChatOrchestrator] | None = None,
    selection_func: GroupChatSelectionFunction | None = None,
    orchestrator_name: str | None = None,
    # Optional tuning
    termination_condition: TerminationCondition | None = None,
    max_rounds: int | None = None,
    checkpoint_storage: CheckpointStorage | None = None,
    output_from: Sequence[...] | "all" | None = ...,
    intermediate_output_from: ... = None,
)
```

| Parameter | Notes |
|---|---|
| `participants` | Agent instances or custom `Executor` nodes. Each must have a unique `name` / `id`. |
| `participant_factories` | Callables returning instances — evaluated at `build()` time. Mutually exclusive with `participants`. |
| `selection_func` | `(GroupChatState) -> str \| None` or async. Returns the name of the next participant, or `None` to terminate the chat early. |
| `orchestrator_agent` | An `Agent` that produces an `AgentOrchestrationOutput` (JSON structured output). |
| `orchestrator` | A fully constructed `BaseGroupChatOrchestrator` instance. |
| `termination_condition` | `(list[Message]) -> bool` or `async (list[Message]) -> bool`. Return `True` to halt. |
| `max_rounds` | Hard cap on selection rounds. |
| `checkpoint_storage` | `FileCheckpointStorage` or `InMemoryCheckpointStorage` for pause/resume. |
| `output_from` | Which participant(s) emit workflow `output` events. Default: orchestrator only. |

### Fluent setter methods

| Method | Returns | Notes |
|---|---|---|
| `.with_termination_condition(cond)` | `Self` | Override `termination_condition`. |
| `.with_max_rounds(n)` | `Self` | Override `max_rounds`. |
| `.with_checkpointing(storage)` | `Self` | Attach checkpoint storage. |
| `.with_request_info(*, agents=None)` | `Self` | Enable human-in-the-loop after each *agent* participant turn. `agents=None` targets all `Agent` participants; custom `Executor` participants are not covered and must handle request info themselves. |
| `.build()` | `Workflow` | Validate and freeze the workflow graph. |

### `GroupChatState` (passed to `selection_func`)

```python
@dataclass(frozen=True)
class GroupChatState:
    current_round: int                       # starts at 0
    participants: OrderedDict[str, str]      # name → description
    conversation: list[Message]              # full history so far
```

### Example 1 — round-robin selection function

```python
import asyncio
from itertools import cycle
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import GroupChatBuilder, GroupChatState

def make_agent(name: str, persona: str) -> Agent:
    return Agent(
        name=name,
        client=OpenAIChatClient(),
        instructions=f"You are {name}. {persona}",
    )

alice = make_agent("Alice", "A skeptical scientist.")
bob   = make_agent("Bob",   "An optimistic entrepreneur.")

# Alternate between Alice and Bob for up to 4 rounds
_participants = cycle(["Alice", "Bob"])
def round_robin(state: GroupChatState) -> str:
    return next(_participants)

workflow = GroupChatBuilder(
    participants=[alice, bob],
    selection_func=round_robin,
    max_rounds=4,
).build()

async def main():
    result = await workflow.run("Should AI replace human creativity?")
    for response in result.get_outputs():
        for message in response.messages:
            print(f"[{message.author_name}] {message.text}")

asyncio.run(main())
```

### Example 2 — LLM-driven orchestrator agent

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import GroupChatBuilder

writer    = Agent(name="Writer",   client=OpenAIChatClient(), instructions="You write first drafts.")
editor    = Agent(name="Editor",   client=OpenAIChatClient(), instructions="You refine and critique drafts.")
publisher = Agent(name="Publisher",client=OpenAIChatClient(), instructions="You approve or reject content.")

orchestrator = Agent(
    name="Director",
    client=OpenAIChatClient(),
    instructions=(
        "You manage a creative writing pipeline. "
        "Route between Writer, Editor, and Publisher as appropriate. "
        "Terminate when the Publisher approves the piece."
    ),
)

workflow = GroupChatBuilder(
    participants=[writer, editor, publisher],
    orchestrator_agent=orchestrator,
    max_rounds=10,
).build()

async def main():
    result = await workflow.run("Write a haiku about sunrise.")
    for response in result.get_outputs():
        for msg in response.messages:
            print(f"[{msg.author_name}] {msg.text}")

asyncio.run(main())
```

### Example 3 — Termination condition + checkpointing

```python
import asyncio
from agent_framework import Agent, Message
from agent_framework import InMemoryCheckpointStorage
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import GroupChatBuilder

analyst = Agent(name="Analyst", client=OpenAIChatClient(), instructions="Analyse the data.")
critic  = Agent(name="Critic",  client=OpenAIChatClient(), instructions="Challenge the analysis.")

def terminate_on_agreement(conversation: list[Message]) -> bool:
    """Stop when the last two assistant messages both contain 'agree'."""
    assistant_msgs = [m for m in conversation if m.role == "assistant"]
    if len(assistant_msgs) < 2:
        return False
    return all("agree" in m.text.lower() for m in assistant_msgs[-2:])

storage = InMemoryCheckpointStorage()

def select_next(state) -> str:
    return "Critic" if state.current_round % 2 == 0 else "Analyst"

workflow = (
    GroupChatBuilder(
        participants=[analyst, critic],
        selection_func=select_next,
        termination_condition=terminate_on_agreement,
        max_rounds=20,
    )
    .with_checkpointing(storage)
    .build()
)

async def main():
    # Initial run — checkpoint_id is for resuming a stored checkpoint, not for naming one.
    result = await workflow.run("Evaluate Q3 sales performance.")
    for response in result.get_outputs():
        for msg in response.messages:
            print(f"[{msg.author_name}] {msg.text[:80]}")

    # To resume: retrieve the latest saved checkpoint ID from storage, then run again.
    # workflow_name must match the workflow's configured name (defaults to class/builder name).
    latest = await storage.get_latest(workflow_name="GroupChatWorkflow")
    if latest:
        resumed = await workflow.run(checkpoint_id=latest.checkpoint_id)
        for response in resumed.get_outputs():
            for msg in response.messages:
                print(f"[resumed][{msg.author_name}] {msg.text[:80]}")

asyncio.run(main())
```

### Example 4 — Human-in-the-loop after each participant

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import GroupChatBuilder

researcher = Agent(name="Researcher", client=OpenAIChatClient(), instructions="Research the topic.")
writer     = Agent(name="Writer",     client=OpenAIChatClient(), instructions="Write the summary.")

workflow = (
    GroupChatBuilder(
        participants=[researcher, writer],
        selection_func=lambda state: "Writer" if state.current_round % 2 else "Researcher",
        max_rounds=6,
    )
    .with_request_info()            # pause after every participant turn
    .build()
)

async def main():
    # Workflow.run() does not accept a response_handler callback.
    # HITL pattern: run → collect request_info events → re-run with responses= dict.
    result = await workflow.run("Summarise recent advances in quantum computing.")
    while result.get_request_info_events():
        responses = {
            event.request_id: input(f"[HITL] Guide (round {i}): ")
            for i, event in enumerate(result.get_request_info_events())
        }
        result = await workflow.run(responses=responses)
    for response in result.get_outputs():
        for msg in response.messages:
            print(f"[{msg.author_name}] {msg.text[:80]}")

asyncio.run(main())
```

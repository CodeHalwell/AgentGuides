---
title: "Class deep-dives Vol. 11 — 10 more LangGraph types"
description: "Source-verified deep dives into InjectedState, InjectedStore, MessagesState, Overwrite, ToolOutputMixin, CheckpointMetadata, CheckpointTuple, StateUpdate, PersistentDict, and DeltaChannelHistory — with runnable examples for every feature."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 11"
  order: 42
---

# Class deep-dives Vol. 11 — 10 more LangGraph types

Verified against **`langgraph==1.2.4`** / **`langgraph-prebuilt==1.1.0`** / **`langgraph-checkpoint==4.1.1`**.

Every section was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

[→ Vol. 1 covers StateGraph, CompiledStateGraph, InMemorySaver, ToolNode, create_react_agent, Command, Send, @task/@entrypoint, BinaryOperatorAggregate/Topic, InMemoryStore](./langgraph_class_deep_dives/)

[→ Vol. 2 covers RetryPolicy, CachePolicy/InMemoryCache, TimeoutPolicy, add_messages/MessagesState, tools_condition, ToolCallTransformer/ToolCallStream, StateSnapshot, IsLastStep/RemainingSteps, ToolRuntime, Runtime/RunControl](./langgraph_class_deep_dives_v2/)

[→ Vol. 3 covers interrupt()/Interrupt, DeltaChannel, EphemeralValue, NamedBarrierValue, RemoveMessage/push_message, Pregel, NodeBuilder, GraphOutput, PregelTask, IndexConfig/TTLConfig](./langgraph_class_deep_dives_v3/)

[→ Vol. 4 covers set_node_defaults, add_sequence, input_schema/output_schema, context_schema/Runtime.context, get_stream_writer/StreamWriter, push_ui_message, entrypoint.final, REMOVE_ALL_MESSAGES, error_handler on add_node, error taxonomy](./langgraph_class_deep_dives_v4/)

[→ Vol. 5 covers RedisCache, EncryptedSerializer, JsonPlusSerializer, UntrackedValue, AnyValue, EmbeddingsLambda/ensure_embeddings, BaseCheckpointSaver, typed StreamParts, task.clear_cache, HumanInterrupt protocol](./langgraph_class_deep_dives_v5/)

[→ Vol. 6 covers GraphRunStream/AsyncGraphRunStream, StreamTransformer, StreamChannel, ValuesTransformer/CustomTransformer/UpdatesTransformer, GraphCallbackHandler, GraphInterruptEvent/GraphResumeEvent, GraphDrained, NodeTimeoutError, delete_ui_message/ui_message_reducer, ProtocolEvent](./langgraph_class_deep_dives_v6/)

[→ Vol. 7 covers PregelProtocol/StreamProtocol, BackgroundExecutor/AsyncBackgroundExecutor, AsyncBatchedBaseStore/_dedupe_ops, get_text_at_path/tokenize_path, SerdeEvent/register_serde_event_listener, BaseChannel, call()/SyncAsyncFuture, PregelScratchpad, StateNodeSpec/node Protocols, identifier/get_runnable_for_task](./langgraph_class_deep_dives_v7/)

[→ Vol. 8 covers ExecutionInfo/Runtime.heartbeat, ServerInfo/BaseUser, ReplayState, StreamMux, Call (functional API internals), ChannelWrite/ChannelWriteEntry, PregelRunner/FuturesDict, WritesProtocol/PregelTaskWrites, SyncPregelLoop/AsyncPregelLoop, DuplexStream](./langgraph_class_deep_dives_v8/)

[→ Vol. 9 covers ToolCallRequest/override(), Send+timeout, create_react_agent pre/post hooks, RetryPolicy chained policies, CachePolicy custom key_func, InMemoryStore raw embeddings, context_schema+Runtime.context, Command.PARENT cross-subgraph routing, TimeoutPolicy.coerce(), entrypoint multi-policy retry](./langgraph_class_deep_dives_v9/)

[→ Vol. 10 covers Durability checkpoint modes, NodeError/NodeCancelledError, TaskPayload/TaskResultPayload, CheckpointPayload/CheckpointTask, Item/SearchItem, GetOp/PutOp/SearchOp/ListNamespacesOp/MatchCondition, UIMessage/RemoveUIMessage, GraphOutput v2, StreamPart variants, PregelExecutableTask/CacheKey](./langgraph_class_deep_dives_v10/)

---

## 1 · `InjectedState` — inject graph state into tools

**Module:** `langgraph.prebuilt`  
**Exported as:** `from langgraph.prebuilt import InjectedState`

`InjectedState` is an annotation you apply to tool function parameters. It signals to `ToolNode` that the parameter should be filled with the current graph state rather than a value provided by the language model. The annotated parameter is **completely invisible** to the LLM — it never appears in the tool's JSON schema.

### Source (1.2.4)

```python
class InjectedState(InjectedToolArg):
    def __init__(self, field: str | None = None) -> None:
        self.field = field
```

`InjectedToolArg` (from `langchain_core`) is the base marker class. `ToolNode` strips any parameter annotated with a subclass of `InjectedToolArg` from the tool schema it presents to the model.

### Key behaviours

| `InjectedState(field)` | What gets injected |
|---|---|
| `InjectedState()` or `InjectedState(None)` | The entire state dict |
| `InjectedState("messages")` | `state["messages"]` |
| `InjectedState("user_id")` | `state["user_id"]` |

### Example 1: Inject full state

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str
    session_id: str

@tool
def personalised_greeting(
    greeting_style: str,
    state: Annotated[dict, InjectedState()],  # receives full state dict
) -> str:
    """Generate a personalised greeting for the current user."""
    name = state.get("user_name", "stranger")
    session = state.get("session_id", "unknown")
    return f"{greeting_style} greeting for {name} (session: {session})"

# The LLM schema for personalised_greeting only has: greeting_style
# 'state' is stripped out entirely — the model never sees it.
```

### Example 2: Inject a specific field

```python
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool

@tool
def check_balance(
    account_type: str,
    user_id: Annotated[str, InjectedState("user_id")],  # only state["user_id"]
) -> str:
    """Check account balance for the current user."""
    # user_id is automatically populated from state["user_id"]
    return f"Balance for user {user_id}, account: {account_type}: $1,234.56"
```

### Example 3: Full working graph with injection

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

class ShoppingState(TypedDict):
    messages: Annotated[list, add_messages]
    cart: list[str]
    user_tier: str  # "standard" | "premium"

@tool
def add_to_cart(
    product_name: str,
    cart: Annotated[list, InjectedState("cart")],
    user_tier: Annotated[str, InjectedState("user_tier")],
) -> str:
    """Add a product to the shopping cart."""
    discount = "20% premium discount" if user_tier == "premium" else "no discount"
    return f"Added '{product_name}' to cart. Current items: {len(cart) + 1}. Applied: {discount}"

@tool
def view_cart(
    state: Annotated[dict, InjectedState()],
) -> str:
    """View current cart contents."""
    cart = state.get("cart", [])
    tier = state.get("user_tier", "standard")
    return f"Cart ({len(cart)} items) for {tier} user: {cart}"

tools = [add_to_cart, view_cart]

# Build graph
def agent_node(state: ShoppingState) -> dict:
    # model logic here; returns AIMessage with tool_calls
    pass

builder = StateGraph(ShoppingState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

result = graph.invoke({
    "messages": [HumanMessage("Add some headphones to my cart")],
    "cart": ["laptop"],
    "user_tier": "premium",
})
```

### Mixing injected and regular args

A tool can freely mix regular model-facing args with any number of injected ones. `ToolNode` handles the split: the LLM fills the regular args, and the injected ones are populated from state at execution time.

```python
@tool
def complex_action(
    # Model fills these:
    action: str,
    quantity: int,
    # ToolNode fills these from state:
    user_id: Annotated[str, InjectedState("user_id")],
    full_state: Annotated[dict, InjectedState()],
) -> str:
    """Perform complex action with state awareness."""
    return f"Action '{action}' x{quantity} for user {user_id}"
```

---

## 2 · `InjectedStore` — inject the persistent store into tools

**Module:** `langgraph.prebuilt`  
**Exported as:** `from langgraph.prebuilt import InjectedStore`

`InjectedStore` is the store-variant of `InjectedState`. Annotate a tool parameter with it and `ToolNode` will inject the `BaseStore` instance that the graph was compiled with. Like `InjectedState`, the parameter is invisible to the LLM.

### Source (1.2.4)

```python
class InjectedStore(InjectedToolArg):
    pass  # no __init__ — just a marker with no configuration
```

### Requirement

The graph must be compiled with a `store=` argument:

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
graph = builder.compile(store=store)
```

### Example 1: Persistent user preferences

```python
from typing import Any, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedStore, tools_condition
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool

class AssistantState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str

@tool
def save_preference(
    key: str,
    value: str,
    store: Annotated[Any, InjectedStore()],
) -> str:
    """Save a user preference to persistent storage."""
    # InMemoryStore.put(namespace, key, value_dict)
    store.put(("preferences", "user_42"), key, {"value": value})
    return f"Saved preference: {key} = {value}"

@tool
def get_preference(
    key: str,
    store: Annotated[Any, InjectedStore()],
) -> str:
    """Retrieve a user preference from persistent storage."""
    item = store.get(("preferences", "user_42"), key)
    if item is None:
        return f"No preference found for '{key}'"
    return f"Preference '{key}' = {item.value['value']}"

tools = [save_preference, get_preference]

store = InMemoryStore()
builder = StateGraph(AssistantState)
builder.add_node("tools", ToolNode(tools))
# ... add agent node, edges
graph = builder.compile(store=store)
```

### Example 2: Cross-session memory with namespaced storage

```python
from typing import Any, Annotated
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool

@tool
def remember_fact(
    fact: str,
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[Any, InjectedStore()],
) -> str:
    """Store a fact about the user for future sessions."""
    import uuid
    fact_id = str(uuid.uuid4())[:8]
    store.put(("memory", user_id), fact_id, {"fact": fact})
    return f"Remembered: {fact}"

@tool
def recall_facts(
    topic: str,
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[Any, InjectedStore()],
) -> str:
    """Recall stored facts about the user relevant to a topic."""
    # search() does semantic/filter search over the namespace
    results = store.search(("memory", user_id), query=topic, limit=5)
    if not results:
        return "No relevant memories found."
    facts = [r.value["fact"] for r in results]
    return f"Recalled {len(facts)} facts: {facts}"
```

### Example 3: Async store access

`InjectedStore` works identically with async nodes. The store's `aput` / `aget` methods are available for full async operation:

```python
@tool
async def async_save(
    key: str,
    value: str,
    store: Annotated[Any, InjectedStore()],
) -> str:
    """Async save to the store."""
    await store.aput(("data",), key, {"v": value})
    return f"Saved {key}"
```

---

## 3 · `MessagesState` — the built-in message-accumulating TypedDict

**Module:** `langgraph.graph`  
**Exported as:** `from langgraph.graph import MessagesState`

`MessagesState` is a one-field `TypedDict` that ships with LangGraph as the canonical starting point for message-driven graphs.

### Source (1.2.4)

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

That is the entire class. It bundles `add_messages` as the reducer for the `messages` field, so you get deduplication, in-place updates by message ID, and `RemoveMessage` support for free.

### When to use it vs a custom TypedDict

| Situation | Recommendation |
|---|---|
| Pure chat agent, no extra state | Use `MessagesState` directly |
| Chat + extra fields (user ID, turn count, …) | Subclass `MessagesState` |
| Non-chat graph, messages is secondary | Define your own TypedDict |

### Example 1: Simplest possible agent

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage

def chat_node(state: MessagesState) -> dict:
    # state["messages"] is always a list[AnyMessage]
    last = state["messages"][-1]
    response = AIMessage(content=f"Echo: {last.content}")
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile()
result = graph.invoke({"messages": [HumanMessage("Hello")]})
print(result["messages"][-1].content)  # "Echo: Hello"
```

### Example 2: Subclassing to add custom fields

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState, StateGraph, START, END

class ConversationState(MessagesState):
    """MessagesState + per-thread metadata."""
    turn_count: int
    user_name: str
    system_prompt: str

def increment_turn(state: ConversationState) -> dict:
    return {"turn_count": state["turn_count"] + 1}

def chat(state: ConversationState) -> dict:
    from langchain_core.messages import AIMessage, SystemMessage
    # Access both messages and custom fields
    name = state.get("user_name", "user")
    system = state.get("system_prompt", "You are a helpful assistant.")
    response = AIMessage(content=f"Hello {name}, I'm on turn {state['turn_count']}")
    return {"messages": [response]}

builder = StateGraph(ConversationState)
builder.add_node("increment", increment_turn)
builder.add_node("chat", chat)
builder.add_edge(START, "increment")
builder.add_edge("increment", "chat")
builder.add_edge("chat", END)

graph = builder.compile()
result = graph.invoke({
    "messages": [{"role": "user", "content": "Hi"}],
    "turn_count": 0,
    "user_name": "Alice",
    "system_prompt": "Be concise.",
})
print(result["turn_count"])  # 1
```

### Example 3: `add_messages` reducer behaviour

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage

# 1. Append new messages
existing = [HumanMessage(content="Hi", id="msg-1")]
new_msg = AIMessage(content="Hello!", id="msg-2")
result = add_messages(existing, [new_msg])
# → [HumanMessage("Hi"), AIMessage("Hello!")]

# 2. Update in-place by matching ID
update = AIMessage(content="Hello, updated!", id="msg-2")
result = add_messages(result, [update])
# → [HumanMessage("Hi"), AIMessage("Hello, updated!")]

# 3. Delete by ID via RemoveMessage
remove = RemoveMessage(id="msg-1")
result = add_messages(result, [remove])
# → [AIMessage("Hello, updated!")]
```

---

## 4 · `Overwrite` — bypass a reducer and write directly

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import Overwrite`

`Overwrite` is a dataclass wrapper that lets you **replace** the entire value of a `BinaryOperatorAggregate` channel in one super-step, ignoring the reducer. Without `Overwrite`, every write goes through the reducer (e.g. `operator.add` appends). With it, the channel value is simply replaced.

### Source (1.2.4)

```python
@dataclass(slots=True)
class Overwrite:
    value: Any
```

### Rules

- Multiple `Overwrite` values targeting the same channel in a single super-step raise `InvalidUpdateError` — only one node may overwrite per step.
- The reducer is not called at all; the stored value becomes exactly `Overwrite.value`.
- Plain writes (non-`Overwrite`) and `Overwrite` writes cannot coexist for the same channel in the same super-step.

### Example 1: Replace an accumulating list

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite

class LogState(TypedDict):
    events: Annotated[list[str], operator.add]  # accumulates by default

def add_event(state: LogState) -> dict:
    return {"events": ["event-A"]}  # appended via operator.add

def reset_log(state: LogState) -> dict:
    # Replaces the entire list, skipping operator.add
    return {"events": Overwrite(value=["fresh-start"])}

builder = StateGraph(LogState)
builder.add_node("add", add_event)
builder.add_node("reset", reset_log)
builder.add_edge(START, "add")
builder.add_edge("add", "reset")
builder.add_edge("reset", END)

graph = builder.compile()
result = graph.invoke({"events": ["initial"]})
print(result["events"])  # ["fresh-start"] — not ["initial", "event-A", "fresh-start"]
```

### Example 2: Conditional reset vs accumulate

```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite

class PipelineState(TypedDict):
    items: Annotated[list[str], operator.add]
    mode: str

def accumulate(state: PipelineState) -> dict:
    return {"items": ["item-from-accumulate"]}

def reset_or_accumulate(state: PipelineState) -> dict:
    if state["mode"] == "reset":
        return {"items": Overwrite(value=[])}  # wipe the list
    return {"items": ["item-from-other"]}  # normal accumulation

def route(state: PipelineState) -> Literal["accumulate", "other"]:
    return "accumulate" if state["mode"] == "accumulate" else "other"

builder = StateGraph(PipelineState)
builder.add_node("accumulate", accumulate)
builder.add_node("other", reset_or_accumulate)
builder.add_edge(START, "accumulate")
builder.add_conditional_edges("accumulate", route)
builder.add_edge("other", END)

graph = builder.compile()

# Accumulate mode
r1 = graph.invoke({"items": ["seed"], "mode": "accumulate"})
print(r1["items"])  # ["seed", "item-from-accumulate"]

# Reset mode
r2 = graph.invoke({"items": ["seed"], "mode": "reset"})
print(r2["items"])  # []
```

### When to use `Overwrite`

- **Periodic cache invalidation**: a node that refreshes a cached list should overwrite, not append.
- **Round-robin slot patterns**: replace the single "current item" instead of appending to it.
- **Error recovery**: a recovery node that needs to hard-reset accumulated errors or retries.

---

## 5 · `ToolOutputMixin` — marker for custom tool return types

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import ToolOutputMixin`

`ToolOutputMixin` is an empty marker mixin class. When a `BaseTool` is invoked with a `ToolCall` and returns an object, `ToolNode` checks if it is a `Command`, a `ToolMessage`, or a list of those. Any other return value is coerced to a string and wrapped in a `ToolMessage`.

`ToolOutputMixin` exists as the **hook point** for future custom types. If you inherit from it, your objects will be recognised as deliberate, structured tool outputs — this matters if LangGraph extends the recognised type list in a later release.

### Source (1.2.4)

```python
class ToolOutputMixin:
    """Mixin for objects that tools can return directly.

    If a custom BaseTool is invoked with a ToolCall and the output of custom code is
    not an instance of ToolOutputMixin, the output will automatically be coerced to
    a string and wrapped in a ToolMessage.
    """
```

### Current valid tool return types

`ToolNode` recognises these return types directly, no coercion needed:

| Return type | Behaviour |
|---|---|
| `ToolMessage` | Used as-is (content coerced with `msg_content_output`) |
| `Command` | Routed via the graph; can carry state updates |
| `list[ToolMessage \| Command]` | Processed element-by-element |
| Anything else | `str(value)` → wrapped in a new `ToolMessage` |

### Example 1: Returning a `ToolMessage` directly for full control

```python
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

@tool
def structured_result(query: str) -> ToolMessage:
    """Return a ToolMessage with custom metadata."""
    return ToolMessage(
        content=f"Result for: {query}",
        tool_call_id="",  # ToolNode fills this in automatically
        additional_kwargs={"source": "internal_db", "confidence": 0.95},
    )
```

### Example 2: Returning a `Command` to update state from a tool

```python
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage

@tool
def approve_request(
    request_id: str,
    cart: Annotated[list, InjectedState("cart")],
) -> Command:
    """Approve a request and update graph state."""
    updated_cart = [item for item in cart if item["id"] != request_id]
    return Command(
        update={
            "cart": updated_cart,
            "last_approved": request_id,
        },
        # Optionally navigate: goto="next_node"
    )
```

### Example 3: Building a `ToolOutputMixin` subclass (future-proofing)

```python
from langgraph.types import ToolOutputMixin

class RichToolResult(ToolOutputMixin):
    """A structured result object that tools can return."""
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata

    def __str__(self) -> str:
        return self.content  # ToolNode coerces to str if not yet natively recognised

@tool
def fetch_data(query: str) -> RichToolResult:
    """Fetch structured data."""
    return RichToolResult(
        content=f"Data for: {query}",
        metadata={"rows": 42, "source": "warehouse"},
    )
```

---

## 6 · `CheckpointMetadata` — metadata stored alongside every checkpoint

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import CheckpointMetadata`

`CheckpointMetadata` is the `TypedDict` that rides alongside every checkpoint stored by a `BaseCheckpointSaver`. It describes *why* and *when* the checkpoint was created.

### Source (1.2.4)

```python
class CheckpointMetadata(TypedDict, total=False):
    source: Literal["input", "loop", "update", "fork"]
    step: int
    parents: dict[str, str]
    run_id: str
    counters_since_delta_snapshot: dict[str, tuple[int, int]]
```

All fields are optional (`total=False`). In practice LangGraph always sets `source`, `step`, and `parents` on normal checkpoints.

### Field reference

| Field | Type | Meaning |
|---|---|---|
| `source` | `"input" \| "loop" \| "update" \| "fork"` | What triggered this checkpoint |
| `step` | `int` | `-1` = input checkpoint, `0` = first loop checkpoint, `n` = nth checkpoint |
| `parents` | `dict[str, str]` | namespace → parent checkpoint ID mapping |
| `run_id` | `str` | ID of the run that produced this checkpoint |
| `counters_since_delta_snapshot` | `dict[str, tuple[int, int]]` | Beta — delta channel bookkeeping |

### `source` values

- **`"input"`**: Created from `invoke()`/`stream()` input. Always at `step=-1`.
- **`"loop"`**: Created automatically at the end of each super-step inside the Pregel loop.
- **`"update"`**: Created by a manual `graph.update_state()` call.
- **`"fork"`**: Created when you replay from a past checkpoint (time-travel).

### Example 1: Reading metadata from a checkpoint

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    value: int

def increment(state: State) -> dict:
    return {"value": state["value"] + 1}

saver = InMemorySaver()
builder = StateGraph(State)
builder.add_node("inc", increment)
builder.add_edge(START, "inc")
builder.add_edge("inc", END)
graph = builder.compile(checkpointer=saver)

config = {"configurable": {"thread_id": "t1"}}
graph.invoke({"value": 0}, config)

# Inspect every checkpoint for this thread
for cp_tuple in saver.list(config):
    meta = cp_tuple.metadata
    print(f"source={meta.get('source')!r:8}  step={meta.get('step'):3}  "
          f"checkpoint_id={cp_tuple.config['configurable']['checkpoint_id'][:8]}")
# source='input'   step= -1  checkpoint_id=...
# source='loop'    step=  0  checkpoint_id=...
```

### Example 2: Filtering checkpoints by metadata

```python
# List only "loop" checkpoints (i.e. those generated during actual execution)
loop_checkpoints = list(saver.list(config, filter={"source": "loop"}))

# List checkpoints at exactly step 0
step_zero = list(saver.list(config, filter={"step": 0}))
```

### Example 3: Metadata on manual update

```python
# After update_state, a new "update" checkpoint is created
new_config = graph.update_state(config, {"value": 99})

for cp_tuple in saver.list(new_config):
    print(cp_tuple.metadata.get("source"), cp_tuple.metadata.get("step"))
# update  1   ← the injected step from update_state
# loop    0
# input  -1
```

---

## 7 · `CheckpointTuple` — the complete checkpoint snapshot

**Module:** `langgraph.checkpoint.base`  
**Exported as:** `from langgraph.checkpoint.base import CheckpointTuple`

`CheckpointTuple` is a `NamedTuple` that wraps all data associated with a single checkpoint. It is the return type of `BaseCheckpointSaver.get_tuple()` and the element type yielded by `list()`.

### Source (1.2.4)

```python
class CheckpointTuple(NamedTuple):
    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: RunnableConfig | None = None
    pending_writes: list[PendingWrite] | None = None
```

### Fields

| Field | Type | Description |
|---|---|---|
| `config` | `RunnableConfig` | The config identifying this checkpoint (contains `thread_id`, `checkpoint_ns`, `checkpoint_id`) |
| `checkpoint` | `Checkpoint` | The serialised checkpoint dict with `channel_values`, `channel_versions`, etc. |
| `metadata` | `CheckpointMetadata` | Source, step, parents (see §6) |
| `parent_config` | `RunnableConfig \| None` | Config pointing to the previous checkpoint (enables time-travel) |
| `pending_writes` | `list[PendingWrite] \| None` | Writes that were in-flight when the checkpoint was saved (from interrupted runs) |

### Example 1: Inspecting a checkpoint

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    counter: int
    label: str

def step(state: State) -> dict:
    return {"counter": state["counter"] + 1}

saver = InMemorySaver()
builder = StateGraph(State)
builder.add_node("step", step)
builder.add_edge(START, "step")
builder.add_edge("step", END)
graph = builder.compile(checkpointer=saver)

config = {"configurable": {"thread_id": "inspect-demo"}}
graph.invoke({"counter": 0, "label": "hello"}, config)

# Get the latest checkpoint tuple
cp = saver.get_tuple(config)
print("channel_values:", cp.checkpoint["channel_values"])
# channel_values: {'counter': 1, 'label': 'hello', ...}

print("step:", cp.metadata["step"])        # 0 or 1 depending on graph design
print("source:", cp.metadata["source"])    # "loop"
print("has parent:", cp.parent_config is not None)  # True
```

### Example 2: Walking the checkpoint chain

```python
config = {"configurable": {"thread_id": "walk-demo"}}
graph.invoke({"counter": 0, "label": "start"}, config)
graph.invoke({"counter": 0, "label": "start"}, config)  # second run

all_checkpoints = list(saver.list(config))
print(f"Total checkpoints: {len(all_checkpoints)}")

for i, cp in enumerate(all_checkpoints):
    step = cp.metadata.get("step", "?")
    source = cp.metadata.get("source", "?")
    cid = cp.config["configurable"]["checkpoint_id"][:8]
    pid = (cp.parent_config or {}).get("configurable", {}).get("checkpoint_id", "none")[:8]
    print(f"  [{i}] step={step:3}  source={source!r:8}  id={cid}  parent={pid}")
```

### Example 3: Time-travel using `parent_config`

```python
config = {"configurable": {"thread_id": "time-travel"}}
graph.invoke({"counter": 0, "label": "original"}, config)

# Get all checkpoints ordered newest → oldest
checkpoints = list(saver.list(config))

# Go back to the step before the last one
target = checkpoints[1]  # second newest
past_state = graph.get_state(target.config)
print("Rewound to:", past_state.values)

# Branch from that historical point
result = graph.invoke(
    {"counter": past_state.values["counter"], "label": "branched"},
    target.config,
)
```

### Example 4: Inspecting pending writes (interrupted runs)

```python
from langgraph.types import Interrupt

# With interrupt_before, the checkpoint has pending writes
graph_with_interrupt = builder.compile(
    checkpointer=saver,
    interrupt_before=["step"],
)
config2 = {"configurable": {"thread_id": "interrupted"}}
graph_with_interrupt.invoke({"counter": 0, "label": "paused"}, config2)

cp = saver.get_tuple(config2)
if cp.pending_writes:
    print(f"Pending writes: {len(cp.pending_writes)}")
    for task_id, channel, value in cp.pending_writes:
        print(f"  task={task_id[:8]}  channel={channel!r}  value={value!r}")
```

---

## 8 · `StateUpdate` — structured argument to `update_state` / `bulk_update_state`

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import StateUpdate`

`StateUpdate` is a `NamedTuple` that `update_state()` converts its arguments into before delegating to the lower-level `bulk_update_state()`.

### Source (1.2.4)

```python
class StateUpdate(NamedTuple):
    values: dict[str, Any] | None
    as_node: str | None = None
    task_id: str | None = None
```

And `update_state` is implemented as:

```python
def update_state(self, config, values, as_node=None, task_id=None):
    return self.bulk_update_state(config, [[StateUpdate(values, as_node, task_id)]])
```

### Fields

| Field | Type | Meaning |
|---|---|---|
| `values` | `dict[str, Any] \| None` | State delta to apply (same shape as node output). `None` to skip values but still record. |
| `as_node` | `str \| None` | Pretend this update came from this node. Affects which edges trigger next. If `None`, LangGraph infers the last-updated node. |
| `task_id` | `str \| None` | Associate the update with a specific task ID. Used internally; rarely needed manually. |

### Example 1: Basic `update_state`

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class WorkflowState(TypedDict):
    status: str
    result: str
    retries: int

saver = InMemorySaver()
builder = StateGraph(WorkflowState)
builder.add_node("process", lambda s: {"status": "processing", "retries": s["retries"] + 1})
builder.add_edge(START, "process")
builder.add_edge("process", END)
graph = builder.compile(checkpointer=saver)

config = {"configurable": {"thread_id": "update-demo"}}
graph.invoke({"status": "pending", "result": "", "retries": 0}, config)

# Inject an external result (e.g. from a human reviewer)
new_config = graph.update_state(
    config,
    {"result": "approved", "status": "done"},
    as_node="process",  # make it look like it came from the "process" node
)

# Read back the updated state
state = graph.get_state(new_config)
print(state.values)  # {'status': 'done', 'result': 'approved', 'retries': 1}
```

### Example 2: `bulk_update_state` for multiple updates in one operation

`bulk_update_state` accepts a list of lists of `StateUpdate` objects and applies them atomically in a single checkpoint write:

```python
from langgraph.types import StateUpdate

# Apply two updates in one operation
new_config = graph.bulk_update_state(
    config,
    [
        [StateUpdate({"status": "reviewing"}, as_node="process")],
        [StateUpdate({"result": "approved"}, as_node="process")],
    ],
)
```

### Example 3: Resuming an interrupted graph via `update_state`

```python
from langgraph.types import Command

# Interrupt mid-graph, then resume with human-provided input
graph_interrupted = builder.compile(checkpointer=saver, interrupt_before=["process"])
config = {"configurable": {"thread_id": "human-in-loop"}}
graph_interrupted.invoke({"status": "pending", "result": "", "retries": 0}, config)

# Human reviews and injects a decision
graph_interrupted.update_state(config, {"result": "human says: proceed"})

# Resume — graph continues from where it left off
final = graph_interrupted.invoke(None, config)
print(final["result"])
```

### Example 4: Using `as_node` to influence routing

```python
from typing import Literal

class RouterState(TypedDict):
    decision: str

def router(state: RouterState) -> Literal["path_a", "path_b"]:
    return state["decision"]

builder = StateGraph(RouterState)
builder.add_node("decide", lambda s: s)
builder.add_node("path_a", lambda s: {"decision": "went-a"})
builder.add_node("path_b", lambda s: {"decision": "went-b"})
builder.add_edge(START, "decide")
builder.add_conditional_edges("decide", router)
builder.add_edge("path_a", END)
builder.add_edge("path_b", END)

graph = builder.compile(checkpointer=saver, interrupt_before=["decide"])
config = {"configurable": {"thread_id": "route-demo"}}
graph.invoke({"decision": "path_a"}, config)

# Override the decision before it executes
graph.update_state(config, {"decision": "path_b"}, as_node="decide")
result = graph.invoke(None, config)
print(result["decision"])  # "went-b"
```

---

## 9 · `PersistentDict` — file-backed dictionary for local checkpointing

**Module:** `langgraph.checkpoint.memory`  
**Internal class** — not part of the public API; accessed indirectly via `MemorySaver`

`PersistentDict` is a `defaultdict` subclass that serialises itself to disk using pickle. It powers `MemorySaver` (the file-persisted alias for `InMemorySaver`) when you want checkpoints to survive process restarts.

### Source (1.2.4)

```python
class PersistentDict(defaultdict):
    def __init__(self, *args, filename: str, **kwds):
        self.flag = "c"   # r=readonly, c=create-or-open, n=new (overwrites)
        self.mode = None  # optional chmod octal like 0o644
        self.format = "pickle"
        self.filename = filename
        super().__init__(*args, **kwds)

    def sync(self) -> None:
        """Write dict to disk atomically via a .tmp file."""
        ...

    def close(self) -> None:
        self.sync(); self.clear()
```

### Key properties

| Property | Meaning |
|---|---|
| `filename` | Path to the pickle file |
| `flag = "c"` | Create-or-open (default) |
| `flag = "r"` | Read-only; `sync()` is a no-op |
| `flag = "n"` | Always create new (overwrites existing) |

### Example 1: Direct use for simple persistent storage

```python
from langgraph.checkpoint.memory import PersistentDict
import tempfile, os

with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    filepath = f.name

# Write
with PersistentDict(filename=filepath) as d:
    d["session-001"] = {"messages": ["Hello", "World"], "turn": 3}
    d["session-002"] = {"messages": ["Hi"], "turn": 1}
    # close() calls sync() which flushes to disk atomically

# Read back in a new process
with PersistentDict(filename=filepath) as d:
    session = d["session-001"]
    print(session)  # {'messages': ['Hello', 'World'], 'turn': 3}

os.unlink(filepath)
```

### Example 2: `MemorySaver` with a persistent file

`MemorySaver` is an alias for `InMemorySaver` that accepts a `filepath` to back its storage with `PersistentDict`:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    counter: int

saver = MemorySaver()  # in-memory only; or pass filepath="checkpoints.pkl"

builder = StateGraph(State)
builder.add_node("inc", lambda s: {"counter": s["counter"] + 1})
builder.add_edge(START, "inc")
builder.add_edge("inc", END)
graph = builder.compile(checkpointer=saver)

config = {"configurable": {"thread_id": "persistent-thread"}}
graph.invoke({"counter": 0}, config)
graph.invoke({"counter": 0}, config)  # resumes, counter becomes 2

state = graph.get_state(config)
print(state.values["counter"])  # 2
```

### Example 3: Manual sync control

```python
from langgraph.checkpoint.memory import PersistentDict
import tempfile

filepath = tempfile.mktemp(suffix=".pkl")

# Use without context manager for manual sync control
d = PersistentDict(filename=filepath)
d["key1"] = "value1"
d.sync()          # flush to disk explicitly

d["key2"] = "value2"
# Not synced yet — will be written on close() or next sync()
d.close()         # sync + clear
```

### When to use

Use `PersistentDict` / `MemorySaver` for:

- **Local development**: persist conversation state between script runs without a database.
- **Testing**: reproducible multi-turn test cases that don't reset on re-run.
- **Single-process scripts**: lightweight apps where PostgreSQL/Redis is overkill.

For production multi-process deployments, use `PostgresSaver` or `AsyncPostgresSaver` from `langgraph-checkpoint-postgres`.

---

## 10 · `DeltaChannelHistory` — per-channel write history for delta channels

**Module:** `langgraph.checkpoint.base`  
**Status:** **Beta** — field names and semantics may change

`DeltaChannelHistory` is the `TypedDict` returned per-channel by `BaseCheckpointSaver.get_delta_channel_history()`. It exposes the full write history that a `DeltaChannel` needs to reconstruct its current value by replaying deltas from a snapshot seed.

### Source (1.2.4)

```python
class DeltaChannelHistory(TypedDict):
    writes: list[PendingWrite]
    seed: NotRequired[Any]
```

### Fields

| Field | Type | Presence | Meaning |
|---|---|---|---|
| `writes` | `list[PendingWrite]` | Always | Ordered list of writes oldest→newest for this channel, excluding the target checkpoint's own pending writes |
| `seed` | `Any` | Optional | The stored channel value at the nearest ancestor checkpoint. Absent if no stored value was found (channel starts empty). A `_DeltaSnapshot` blob for true delta channels; a plain value for migrated channels. |

`PendingWrite` is a 3-tuple `(task_id: str, channel: str, value: Any)`.

### Example 1: Reading delta channel history

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Interrupt
from typing_extensions import TypedDict
import operator
from typing import Annotated

class LogState(TypedDict):
    log: Annotated[list[str], operator.add]  # accumulates

saver = InMemorySaver()
builder = StateGraph(LogState)
builder.add_node("append", lambda s: {"log": [f"step-{len(s['log'])+1}"]})
builder.add_edge(START, "append")
builder.add_edge("append", END)
graph = builder.compile(checkpointer=saver)

config = {"configurable": {"thread_id": "delta-history-demo"}}
graph.invoke({"log": []}, config)
graph.invoke({"log": []}, config)  # second invocation continues thread

# Retrieve the delta channel history for the "log" channel
# (InMemorySaver implements get_delta_channel_history)
history = saver.get_delta_channel_history(config=config, channels=["log"])
log_history = history.get("log", {})

print("Writes:", log_history.get("writes"))
# [('task-id-1', 'log', ['step-1']), ('task-id-2', 'log', ['step-2'])]

if "seed" in log_history:
    print("Seed:", log_history["seed"])
```

### Example 2: Async variant

```python
import asyncio

async def read_history():
    history = await saver.aget_delta_channel_history(
        config=config,
        channels=["log"],
    )
    return history

asyncio.run(read_history())
```

### When `seed` is absent vs. present

| Condition | `seed` presence |
|---|---|
| Channel has no stored ancestor blob | Absent — consumer reconstructs from empty start |
| Channel has a `_DeltaSnapshot` ancestor | Present as `_DeltaSnapshot` — replay deltas on top |
| Pre-delta plain-value ancestor | Present as plain value — represents the full value at that point |

### Relationship to `DeltaChannel`

`DeltaChannel` (from Vol. 3) stores only incremental deltas instead of full values. It calls `get_delta_channel_history()` to walk up the parent chain when reconstructing its current value. `DeltaChannelHistory` is the per-channel result of that walk — you consume it if you are building a custom checkpointer or debugging delta channel reconstruction.

---

## Summary table

| Class | Module | Purpose |
|---|---|---|
| `InjectedState` | `langgraph.prebuilt` | Annotate tool params to receive graph state automatically |
| `InjectedStore` | `langgraph.prebuilt` | Annotate tool params to receive the compiled store |
| `MessagesState` | `langgraph.graph` | One-field TypedDict with `add_messages` reducer built in |
| `Overwrite` | `langgraph.types` | Wrap a value to bypass the reducer and replace the channel directly |
| `ToolOutputMixin` | `langgraph.types` | Marker mixin for custom objects returned from tools |
| `CheckpointMetadata` | `langgraph.types` | TypedDict: source, step, parents, run_id per checkpoint |
| `CheckpointTuple` | `langgraph.checkpoint.base` | NamedTuple: config + checkpoint + metadata + parent + pending writes |
| `StateUpdate` | `langgraph.types` | NamedTuple used by `update_state` / `bulk_update_state` |
| `PersistentDict` | `langgraph.checkpoint.memory` | File-backed defaultdict powering `MemorySaver` |
| `DeltaChannelHistory` | `langgraph.checkpoint.base` | Beta TypedDict: writes + optional seed for delta channel reconstruction |

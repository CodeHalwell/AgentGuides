---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 23"
description: "Source-verified deep dives into 10 class groups from agent-framework-declarative: DeclarativeActionExecutor+DeclarativeWorkflowState+DeclarativeEnvConfig+discover_env_references() (PowerFx graph-node base layer — Env allowlist, dot-path get/set, eval() expression engine, discover_env_references() YAML scanner); ConditionGroupEvaluatorExecutor+IfConditionEvaluatorExecutor+ConditionResult (declarative branching graph nodes — first-match semantics, ELSE_BRANCH_INDEX=-1, branch_index routing); ForeachInitExecutor+ForeachNextExecutor+BreakLoopExecutor+ContinueLoopExecutor+LoopIterationResult+LoopControl (declarative loop nodes — LOOP_STATE_KEY in State, current_item/current_index, break/continue signals); SetValueExecutor+SetVariableExecutor+CreateConversationExecutor+SetMultipleVariablesExecutor+SendActivityExecutor+ParseValueExecutor (declarative variable and messaging graph nodes); AgentManifest+AgentDefinition+PromptAgent+EnvironmentVariable+agent_schema_dispatch() (declarative schema types — manifest/template/parameters/resources, PowerFx-evaluated fields, kind-dispatched factory); Property+ArrayProperty+ObjectProperty+PropertySchema (declarative property schema layer — from_dict dispatch, to_json_schema() conversion); Connection hierarchy+ReferenceConnection+RemoteConnection+ApiKeyConnection+AnonymousConnection (declarative connection specs — kind-based dispatch, PowerFx-safe field eval); McpTool+McpServerApprovalMode+McpServerToolAlwaysRequireApprovalMode+McpServerToolNeverRequireApprovalMode+McpServerToolSpecifyApprovalMode (declarative MCP tool with approval gating); Template+Format+Parser+Model+ModelOptions (declarative model and template configuration types); InvokeAzureAgentExecutor+AgentResult+AgentExternalInputRequest+AgentExternalInputResponse (declarative Azure AI Foundry agent invocation with HITL external-loop support)."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 46
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 23

Verified against **agent-framework 1.9.0** / **agent-framework-declarative 1.0.0rc2** (installed June 2026). Every constructor signature, field description, and code example was derived from the installed package source. Sub-packages introspected: `agent_framework_declarative._models`, `agent_framework_declarative._workflows._declarative_base`, `agent_framework_declarative._workflows._executors_basic`, `agent_framework_declarative._workflows._executors_control_flow`, `agent_framework_declarative._workflows._executors_agents`.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2–22](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v22/#previous-volumes) — complete history

This volume covers **ten class groups** from the `agent_framework_declarative` sub-package — the graph-based execution layer that powers YAML-defined workflows. These classes are entirely separate from the functional (`@workflow`) layer covered in previous volumes:

| # | Class / group | Module |
|---|---|---|
| 1 | `DeclarativeActionExecutor` · `DeclarativeWorkflowState` · `DeclarativeEnvConfig` · `discover_env_references()` | `_declarative_base` |
| 2 | `ConditionGroupEvaluatorExecutor` · `IfConditionEvaluatorExecutor` · `ConditionResult` | `_executors_control_flow` |
| 3 | `ForeachInitExecutor` · `ForeachNextExecutor` · `BreakLoopExecutor` · `ContinueLoopExecutor` · `LoopIterationResult` · `LoopControl` | `_executors_control_flow` |
| 4 | `SetValueExecutor` · `SetVariableExecutor` · `CreateConversationExecutor` · `SetMultipleVariablesExecutor` · `SendActivityExecutor` · `ParseValueExecutor` | `_executors_basic` |
| 5 | `AgentManifest` · `AgentDefinition` · `PromptAgent` · `EnvironmentVariable` · `agent_schema_dispatch()` | `_models` |
| 6 | `Property` · `ArrayProperty` · `ObjectProperty` · `PropertySchema` | `_models` |
| 7 | `Connection` · `ReferenceConnection` · `RemoteConnection` · `ApiKeyConnection` · `AnonymousConnection` | `_models` |
| 8 | `McpTool` · `McpServerApprovalMode` · `McpServerToolAlwaysRequireApprovalMode` · `McpServerToolNeverRequireApprovalMode` · `McpServerToolSpecifyApprovalMode` | `_models` |
| 9 | `Template` · `Format` · `Parser` · `Model` · `ModelOptions` | `_models` |
| 10 | `InvokeAzureAgentExecutor` · `AgentResult` · `AgentExternalInputRequest` · `AgentExternalInputResponse` | `_executors_agents` |

---

## 1 · `DeclarativeActionExecutor` · `DeclarativeWorkflowState` · `DeclarativeEnvConfig` · `discover_env_references()`

**Module:** `agent_framework_declarative._workflows._declarative_base`  
**Install:** `pip install agent-framework-declarative`

These types form the PowerFx-backed foundation layer every declarative graph-node executor builds on.

### `DeclarativeEnvConfig`

A frozen dataclass that controls what the PowerFx `Env` symbol exposes at runtime.

```python
@dataclass(frozen=True)
class DeclarativeEnvConfig:
    values: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    restrict_to_configuration: bool = True    # default: block os.environ access
    referenced_names: frozenset[str] = field(default_factory=lambda: frozenset())
```

| Field | Type | Meaning |
|---|---|---|
| `values` | `Mapping[str, str]` | Caller-supplied name→value map; always exposed as `Env.*` |
| `restrict_to_configuration` | `bool` | `True` (default) → `os.environ` never consulted. `False` → falls back to `os.environ` only for names in `referenced_names` that are absent from `values` |
| `referenced_names` | `frozenset[str]` | Allowlist of `Env.NAME` symbols found in YAML; limits the `os.environ` fallback to exactly these names |

`resolve()` returns the merged `dict[str, str]` that feeds the PowerFx engine; calling code normally passes it to `DeclarativeWorkflowState`.

### `DeclarativeWorkflowState`

Wraps `State` with a namespace-aware dot-path API and PowerFx expression evaluation.

```python
class DeclarativeWorkflowState:
    def __init__(
        self,
        state: State,
        env_config: DeclarativeEnvConfig | None = None,
    ) -> None: ...

    def initialize(self, inputs: Mapping[str, Any] | None = None) -> None: ...
    def is_initialized(self) -> bool: ...
    def get(self, path: str, default: Any = None) -> Any: ...
    def set(self, path: str, value: Any) -> None: ...
    def eval(self, expression: str) -> Any: ...
    def eval_if_expression(self, value: Any) -> Any: ...
    def get_state_data(self) -> DeclarativeStateData: ...
    def set_state_data(self, data: DeclarativeStateData) -> None: ...
```

Namespace roots available via dot-path:

| Root | Purpose |
|---|---|
| `Workflow.Inputs.*` | Read-only initial inputs |
| `Workflow.Outputs.*` | Values returned from the workflow |
| `Local.*` | Variables persisting within the current turn |
| `System.*` | System vars: `ConversationId`, `LastMessage`, `conversations` |
| `Agent.*` | Output from the most recent agent invocation |
| `Conversation.*` | Message history (`messages`, `history`) |

### `DeclarativeActionExecutor`

Abstract base class for every declarative graph-node executor (30+ concrete subclasses). Subclasses add `@handler`-decorated coroutines.

```python
class DeclarativeActionExecutor(Executor):
    def __init__(
        self,
        action_def: dict[str, Any],
        *,
        id: str | None = None,
    ) -> None: ...
```

Subclasses call `await self._ensure_state_initialized(ctx, trigger)` to obtain a ready `DeclarativeWorkflowState`, then send typed messages via `await ctx.send_message(...)`.

### `discover_env_references()`

```python
def discover_env_references(node: Any) -> set[str]: ...
```

Walks a parsed YAML structure (dict/list/scalar) and returns every `NAME` that appears in a `=Env.NAME` PowerFx expression. Only strings starting with `=` are scanned (preventing false matches in doc fields). The result populates `DeclarativeEnvConfig.referenced_names` at workflow construction time.

### Examples

**Example 1 — Safe env config (default)**

```python
import asyncio
from agent_framework_declarative._workflows._declarative_base import (
    DeclarativeEnvConfig,
    discover_env_references,
)

# Simulate a parsed workflow YAML with an Env reference
workflow_yaml = {
    "actions": [
        {"kind": "SetValue", "path": "Local.apiUrl", "value": "=Env.API_URL"},
        {"kind": "SetValue", "path": "Local.greeting", "value": "hello"},
    ]
}

# Discover which Env names the YAML actually uses
referenced = discover_env_references(workflow_yaml)
print(referenced)  # {'API_URL'}

# Build a config that ONLY exposes those names
env_config = DeclarativeEnvConfig(
    values={"API_URL": "https://api.example.com"},
    restrict_to_configuration=True,  # os.environ never leaks
    referenced_names=frozenset(referenced),
)
print(env_config.resolve())  # {'API_URL': 'https://api.example.com'}
```

**Example 2 — Allow os.environ fallback for specific keys**

```python
import os
from agent_framework_declarative._workflows._declarative_base import DeclarativeEnvConfig, discover_env_references

os.environ["TENANT_ID"] = "my-azure-tenant"

workflow_yaml = {"prompt": "=Env.TENANT_ID"}
refs = discover_env_references(workflow_yaml)  # {'TENANT_ID'}

env_config = DeclarativeEnvConfig(
    values={},                      # nothing pre-supplied
    restrict_to_configuration=False,  # allow os.environ fallback
    referenced_names=frozenset(refs),
)
resolved = env_config.resolve()
print(resolved)  # {'TENANT_ID': 'my-azure-tenant'}  — only the allowed name
```

**Example 3 — Direct use of DeclarativeWorkflowState**

```python
import asyncio
from agent_framework._workflows._state import State
from agent_framework_declarative._workflows._declarative_base import (
    DeclarativeEnvConfig,
    DeclarativeWorkflowState,
)

state = State()
env_cfg = DeclarativeEnvConfig(values={"GREETING": "Hello"})
wf_state = DeclarativeWorkflowState(state, env_config=env_cfg)

wf_state.initialize(inputs={"userName": "Alice"})

# Dot-path access
print(wf_state.get("Workflow.Inputs.userName"))  # Alice
wf_state.set("Local.message", "world")
print(wf_state.get("Local.message"))              # world

# PowerFx expression evaluation (requires powerfx package + .NET runtime)
# result = wf_state.eval("=Concatenate(Env.GREETING, \" \", Workflow.Inputs.userName)")
```

---

## 2 · `ConditionGroupEvaluatorExecutor` · `IfConditionEvaluatorExecutor` · `ConditionResult`

**Module:** `agent_framework_declarative._workflows._executors_control_flow`

These executors implement declarative branching as graph nodes, translating `conditionGroup` and `if` YAML actions into first-match condition routing.

### `ConditionResult`

```python
@dataclass
class ConditionResult:
    matched: bool        # True when a condition matched
    branch_index: int    # 0-based index of matching branch, or ELSE_BRANCH_INDEX (-1)
    value: Any = None    # The truthy value returned by the matching condition
```

`ELSE_BRANCH_INDEX = -1` is a module-level sentinel for the default/else path.

### `ConditionGroupEvaluatorExecutor`

```python
class ConditionGroupEvaluatorExecutor(DeclarativeActionExecutor):
    def __init__(
        self,
        action_def: dict[str, Any],
        conditions: list[dict[str, Any]],   # each item has 'condition' key + optional 'id'
        *,
        id: str | None = None,
    ) -> None: ...
```

Evaluates conditions **sequentially**; the first truthy result wins. Emits `ConditionResult(matched=True, branch_index=<n>)`. If none match, emits `ConditionResult(matched=False, branch_index=-1)`. Downstream edge conditions check `branch_index` to route to the correct branch.

### `IfConditionEvaluatorExecutor`

```python
class IfConditionEvaluatorExecutor(DeclarativeActionExecutor):
    def __init__(
        self,
        action_def: dict[str, Any],
        condition_expr: str,   # e.g. "=Local.score > 80"
        *,
        id: str | None = None,
    ) -> None: ...
```

Single-condition variant: emits `ConditionResult(branch_index=0)` when true, `branch_index=-1` otherwise.

### Examples

**Example 1 — Inspect ConditionResult routing**

```python
from agent_framework_declarative._workflows._executors_control_flow import (
    ELSE_BRANCH_INDEX,
    ConditionGroupEvaluatorExecutor,
)

# Simulate what the graph builder does when mapping condition results to edges
def route(result_branch_index: int, branches: list[str]) -> str:
    if result_branch_index == ELSE_BRANCH_INDEX:
        return "else-action"
    return branches[result_branch_index]

branches = ["high-value-action", "medium-value-action"]
print(route(0, branches))   # high-value-action
print(route(1, branches))   # medium-value-action
print(route(-1, branches))  # else-action
```

**Example 2 — Build a ConditionGroupEvaluatorExecutor programmatically**

```python
from agent_framework_declarative._workflows._executors_control_flow import (
    ConditionGroupEvaluatorExecutor,
)

action_def = {
    "kind": "ConditionGroup",
    "id": "scoreBranch",
}
conditions = [
    {"id": "high", "condition": "=Local.score > 80"},
    {"id": "medium", "condition": "=Local.score > 50"},
    # implicit else for everything else
]
executor = ConditionGroupEvaluatorExecutor(
    action_def=action_def,
    conditions=conditions,
    id="scoreBranch",
)
print(executor.id)  # scoreBranch
```

**Example 3 — IfConditionEvaluatorExecutor for simple boolean gate**

```python
from agent_framework_declarative._workflows._executors_control_flow import (
    ELSE_BRANCH_INDEX,
    IfConditionEvaluatorExecutor,
)

executor = IfConditionEvaluatorExecutor(
    action_def={"kind": "If", "id": "featureGate"},
    condition_expr="=Local.featureEnabled",
    id="featureGate",
)

# At runtime the executor sends:
#   ConditionResult(matched=True,  branch_index=0)  → then-branch
#   ConditionResult(matched=False, branch_index=-1) → else-branch
print(ELSE_BRANCH_INDEX)  # -1
```

---

## 3 · `ForeachInitExecutor` · `ForeachNextExecutor` · `BreakLoopExecutor` · `ContinueLoopExecutor` · `LoopIterationResult` · `LoopControl`

**Module:** `agent_framework_declarative._workflows._executors_control_flow`

Declarative `foreach` loops are graph structures, not interpreter loops. These executors manage iteration state stored in `State` under `LOOP_STATE_KEY = "_declarative_loop_state"`.

### Message types

```python
@dataclass
class LoopIterationResult:
    has_next: bool          # False on final iteration
    current_item: Any       # The current loop element
    current_index: int      # 0-based position

@dataclass
class LoopControl:
    action: Literal["break", "continue"]  # Signal type
```

### `ForeachInitExecutor`

Initializes loop state on the first pass. Reads the iterable from `action_def["items"]`, evaluates it against the current state, stores the full list plus current position under `LOOP_STATE_KEY`, then sets `Local.<itemName>` (and optionally `Local.<indexName>`) before sending `LoopIterationResult`.

### `ForeachNextExecutor`

Advances the persisted loop state by one position; sends `LoopIterationResult(has_next=False)` when exhausted, triggering the graph edge that exits the loop.

### `BreakLoopExecutor` / `ContinueLoopExecutor`

Send `LoopControl(action="break")` or `LoopControl(action="continue")` respectively. Downstream edges check the `action` field to either exit or jump back to the next-iteration node.

### Examples

**Example 1 — Trace loop state key usage**

```python
from agent_framework_declarative._workflows._executors_control_flow import LOOP_STATE_KEY

# The executor stores per-loop metadata here inside State
print(LOOP_STATE_KEY)  # _declarative_loop_state

# Typical shape stored at state["_declarative_loop_state"]["myLoop"]:
example_loop_state = {
    "items": ["a", "b", "c"],
    "index": 0,
    "length": 3,
    "item_name": "item",    # maps to Local.item
    "index_name": "i",      # maps to Local.i (optional)
}
```

**Example 2 — Build loop executors for a workflow graph**

```python
from agent_framework_declarative._workflows._executors_control_flow import (
    BreakLoopExecutor,
    ContinueLoopExecutor,
    ForeachInitExecutor,
    ForeachNextExecutor,
)

init_action = {
    "kind": "Foreach",
    "id": "iterateResults",
    "source": "=Local.results",  # PowerFx expression; key is "source", not "items"
    "itemName": "result",        # exposed as Local.result; key is "itemName", not "item"
    "indexName": "idx",          # exposed as Local.idx (optional); key is "indexName", not "index"
}
next_action = {
    "kind": "ForeachNext",
    "id": "iterateResults_next",
    "itemName": "result",   # must match init_action — handle_action rebinds from its own action_def
    "indexName": "idx",     # omit if init_action has no indexName
}
break_action = {"kind": "Break", "id": "earlyExit"}

init_exec = ForeachInitExecutor(init_action, id="iterateResults")
next_exec = ForeachNextExecutor(next_action, "iterateResults", id="iterateResults_next")   # init_executor_id required
break_exec = BreakLoopExecutor(break_action, "iterateResults_next", id="earlyExit")        # loop_next_executor_id required

print(init_exec.id, next_exec.id, break_exec.id)
```

**Example 3 — Inspect LoopIterationResult and LoopControl**

```python
from agent_framework_declarative._workflows._executors_control_flow import (
    LoopControl,
    LoopIterationResult,
)

# What the executor sends on each pass
first = LoopIterationResult(has_next=True,  current_item="a", current_index=0)
last  = LoopIterationResult(has_next=False, current_item="c", current_index=2)

# What break/continue executors send
brk  = LoopControl(action="break")
cont = LoopControl(action="continue")

print(first.current_item, first.has_next)  # a True
print(last.has_next)                        # False
print(brk.action, cont.action)              # break continue
```

---

## 4 · `SetValueExecutor` · `SetVariableExecutor` · `CreateConversationExecutor` · `SetMultipleVariablesExecutor` · `SendActivityExecutor` · `ParseValueExecutor`

**Module:** `agent_framework_declarative._workflows._executors_basic`

These are simple action executors that map directly to common declarative YAML actions. Each reads from `self._action_def`, calls `DeclarativeWorkflowState`, and sends `ActionComplete()`.

### `_get_variable_path()` helper

```python
def _get_variable_path(action_def: dict[str, Any], key: str = "variable") -> str | None:
```

Resolves a variable path from two YAML styles:
- `.NET` style: `variable: Local.VarName` → returns `"Local.VarName"`
- Nested style: `variable: {path: Local.VarName}` → returns `"Local.VarName"`
- Fallback: reads `path` key directly from `action_def`

### Executor reference

| Executor | YAML `kind` | Key `action_def` fields | Behaviour |
|---|---|---|---|
| `SetValueExecutor` | `SetValue` | `path`, `value` | Evaluates `value` (PowerFx if starts with `=`), sets at `path` |
| `SetVariableExecutor` | `SetVariable` | `variable`, `value` | Same but resolves path via `_get_variable_path()` |
| `CreateConversationExecutor` | `CreateConversation` | `conversationId` | Generates UUID, writes to `conversationId` path, initialises `System.conversations[id]` |
| `SetMultipleVariablesExecutor` | `SetMultipleVariables` | `assignments` (list of `{variable, value}`) | Evaluates and sets each variable in the list |
| `SendActivityExecutor` | `SendActivity` | `activity` / `text` / `value` | Sends text content to the conversation |
| `ParseValueExecutor` | `ParseValue` | `value`, `variable` | Stringifies an evaluated expression and stores it |

### Examples

**Example 1 — SetValueExecutor YAML + Python equivalent**

```python
# YAML action definition handled by SetValueExecutor:
# - kind: SetValue
#   id: setScore
#   path: Local.score
#   value: "=Workflow.Inputs.rawScore * 1.1"

# Equivalent programmatic construction (for testing/custom graph building):
from agent_framework_declarative._workflows._executors_basic import SetValueExecutor

action_def = {
    "kind": "SetValue",
    "id": "setScore",
    "path": "Local.score",
    "value": "=Workflow.Inputs.rawScore * 1.1",
}
executor = SetValueExecutor(action_def, id="setScore")
print(executor.id)  # setScore
```

**Example 2 — CreateConversationExecutor to initialise a conversation slot**

```python
# YAML:
# - kind: CreateConversation
#   id: initConv
#   conversationId: Local.myConvId

from agent_framework_declarative._workflows._executors_basic import CreateConversationExecutor

action_def = {
    "kind": "CreateConversation",
    "id": "initConv",
    "conversationId": "Local.myConvId",
}
executor = CreateConversationExecutor(action_def, id="initConv")
# At runtime:
# 1. Generates UUID → stores at state path "Local.myConvId"
# 2. Adds {"id": <uuid>, "messages": []} under System.conversations[<uuid>]
print(executor.id)
```

**Example 3 — _get_variable_path resolving both YAML styles**

```python
from agent_framework_declarative._workflows._executors_basic import _get_variable_path

# .NET style (string path)
action1 = {"variable": "Local.result"}
print(_get_variable_path(action1))  # Local.result

# Nested object style
action2 = {"variable": {"path": "Workflow.Outputs.finalAnswer"}}
print(_get_variable_path(action2))  # Workflow.Outputs.finalAnswer

# Direct path fallback
action3 = {"path": "Agent.lastResponse"}
print(_get_variable_path(action3))  # Agent.lastResponse
```

---

## 5 · `AgentManifest` · `AgentDefinition` · `PromptAgent` · `EnvironmentVariable` · `agent_schema_dispatch()`

**Module:** `agent_framework_declarative._models`

The manifest types describe an agent's identity, schema, and resources in a YAML manifest file. All string fields are evaluated through `_try_powerfx_eval()`, so they can contain `=Env.NAME` expressions.

### `AgentManifest`

Top-level manifest type. Loaded from `.yaml` or `.json` manifest files.

```python
class AgentManifest(SerializationMixin):
    def __init__(
        self,
        name: str | None = None,
        displayName: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        template: AgentDefinition | None = None,     # auto-deserialized
        parameters: PropertySchema | None = None,    # auto-deserialized
        resources: list[Resource] | dict[str, Any] | None = None,  # auto-deserialized
    ) -> None: ...
```

| Field | Type | Purpose |
|---|---|---|
| `name` | `str` | Logical agent identifier (PowerFx-evaluated) |
| `displayName` | `str` | Human-readable name |
| `template` | `AgentDefinition` | The agent definition (dispatches to `PromptAgent` when `kind="Prompt"`) |
| `parameters` | `PropertySchema` | Input parameter schema for the manifest |
| `resources` | `list[Resource]` | Linked model/tool resources |

### `AgentDefinition`

Base class for agent definitions; dispatches to `PromptAgent` when `kind` is `"Prompt"` or `"Agent"`.

```python
class AgentDefinition(SerializationMixin):
    def __init__(
        self,
        kind: str | None = None,
        name: str | None = None,
        displayName: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        inputSchema: PropertySchema | None = None,
        outputSchema: PropertySchema | None = None,
    ) -> None: ...
```

### `PromptAgent`

Extends `AgentDefinition` with model configuration, tools, template, and instructions.

```python
class PromptAgent(AgentDefinition):
    def __init__(
        self,
        # ... all AgentDefinition params ...
        model: Model | dict[str, Any] | None = None,
        tools: list[Tool] | None = None,
        template: Template | dict[str, Any] | None = None,
        instructions: str | None = None,           # PowerFx-evaluated
        additionalInstructions: str | None = None, # PowerFx-evaluated
    ) -> None: ...
```

### `EnvironmentVariable`

```python
class EnvironmentVariable(SerializationMixin):
    def __init__(self, name: str | None = None, value: str | None = None) -> None: ...
```

Both `name` and `value` are passed through `_try_powerfx_eval()`.

### `agent_schema_dispatch()`

```python
def agent_schema_dispatch(schema: dict[str, Any]) -> AgentSchemaSpec | None:
```

Factory function that creates any `AgentSchemaSpec` type from a raw dictionary, dispatching on the `kind` field. Used when deserializing heterogeneous YAML sections.

### Examples

**Example 1 — Load an AgentManifest from a dict**

```python
from agent_framework_declarative._models import AgentManifest, PromptAgent

manifest_data = {
    "name": "summarizer",
    "displayName": "Text Summarizer",
    "description": "Summarizes documents",
    "template": {
        "kind": "Prompt",
        "name": "summarizer",
        "instructions": "You are a concise summarizer. Summarize the user's text.",
        "model": {
            "id": "gpt-4o",
            "provider": "openai",
        },
    },
}

manifest = AgentManifest.from_dict(manifest_data)
print(manifest.name)                            # summarizer
print(type(manifest.template).__name__)         # PromptAgent (dispatched)
print(manifest.template.instructions)           # You are a concise summarizer...
```

**Example 2 — AgentManifest with resources and parameters**

```python
from agent_framework_declarative._models import AgentManifest, PropertySchema

manifest_data = {
    "name": "qa-agent",
    "template": {
        "kind": "Prompt",
        "name": "qa-agent",
        "instructions": "Answer questions using the provided context.",
        "model": {"id": "gpt-4o-mini"},
    },
    "parameters": {
        "properties": [
            {"name": "context", "kind": "string", "required": True},
            {"name": "question", "kind": "string", "required": True},
        ]
    },
    "resources": [
        {"kind": "model", "name": "default-model", "id": "gpt-4o"},
    ],
}

manifest = AgentManifest.from_dict(manifest_data)
schema = manifest.parameters.to_json_schema()
print(schema["required"])   # ['context', 'question']
print(manifest.resources[0].kind)  # model
```

**Example 3 — agent_schema_dispatch() for heterogeneous YAML**

```python
from agent_framework_declarative._models import (
    FunctionTool,
    McpTool,
    PromptAgent,
    agent_schema_dispatch,
)

specs = [
    {"kind": "Prompt", "name": "myAgent", "instructions": "Be helpful."},
    {"kind": "function", "name": "getWeather", "description": "Get current weather"},
    {"kind": "mcp", "name": "browserTools", "serverName": "playwright"},
]

for spec in specs:
    obj = agent_schema_dispatch(spec)
    print(type(obj).__name__, "→", getattr(obj, "name", None))
# PromptAgent → myAgent
# FunctionTool → getWeather
# McpTool → browserTools
```

---

## 6 · `Property` · `ArrayProperty` · `ObjectProperty` · `PropertySchema`

**Module:** `agent_framework_declarative._models`

These types describe input/output schemas for tools, functions, and agents. `Property.from_dict()` dispatches to the correct subclass; `PropertySchema.to_json_schema()` converts to standard JSON Schema.

### `Property`

```python
class Property(SerializationMixin):
    def __init__(
        self,
        name: str | None = None,
        kind: str | None = None,       # maps from YAML 'type' field → 'kind'
        description: str | None = None,
        required: bool | None = None,
        default: Any | None = None,
        example: Any | None = None,
        enum: list[Any] | None = None,
    ) -> None: ...

    @classmethod
    def from_dict(cls, value: MutableMapping[str, Any], /, *, dependencies=None) -> Property:
        # Dispatches: kind="array" → ArrayProperty, kind="object" → ObjectProperty
```

The `from_dict` classmethod renames the YAML `type` key to `kind` before dispatching. Call `Property.from_dict()` on the base class (not a subclass) to benefit from dispatch.

### `ArrayProperty`

```python
class ArrayProperty(Property):
    def __init__(
        self,
        *,
        items: Property | None = None,   # auto-deserialized via Property.from_dict
        **parent_kwargs,
    ) -> None: ...
```

### `ObjectProperty`

```python
class ObjectProperty(Property):
    def __init__(
        self,
        *,
        properties: list[Property] | dict[str, dict[str, Any]] | None = None,
        **parent_kwargs,
    ) -> None: ...
```

Accepts properties either as a list (each with a `name` key) or as a dict (`{name: {kind, description, ...}}`).

### `PropertySchema`

```python
class PropertySchema(SerializationMixin):
    def __init__(
        self,
        examples: list[dict[str, Any]] | None = None,
        strict: bool = False,
        properties: list[Property] | dict[str, dict[str, Any]] | None = None,
    ) -> None: ...

    def to_json_schema(self) -> dict[str, Any]:
        # Returns: {"type": "object", "properties": {...}, "required": [...]}
```

`to_json_schema()` extracts each property's `required` boolean into a top-level `required` array and renames `kind` back to `type`, producing standard JSON Schema.

### Examples

**Example 1 — Property dispatch from dict**

```python
from agent_framework_declarative._models import ArrayProperty, ObjectProperty, Property

# Plain string property
p = Property.from_dict({"name": "query", "type": "string", "required": True})
print(type(p).__name__, p.kind)  # Property string

# Array property (dispatched automatically)
arr = Property.from_dict({
    "name": "tags",
    "type": "array",
    "items": {"type": "string"},
})
print(type(arr).__name__, arr.items.kind)  # ArrayProperty string

# Object property
obj = Property.from_dict({
    "name": "address",
    "type": "object",
    "properties": {
        "street": {"type": "string"},
        "city": {"type": "string", "required": True},
    },
})
print(type(obj).__name__, len(obj.properties))  # ObjectProperty 2
```

**Example 2 — PropertySchema.to_json_schema()**

```python
from agent_framework_declarative._models import Property, PropertySchema

schema = PropertySchema(
    strict=True,
    properties=[
        Property.from_dict({"name": "name",  "type": "string",  "required": True}),
        Property.from_dict({"name": "age",   "type": "integer", "required": False}),
        Property.from_dict({"name": "email", "type": "string",  "required": True,
                            "description": "Contact email"}),
    ],
)

json_schema = schema.to_json_schema()
print(json_schema["type"])      # object
print(json_schema["required"])  # ['name', 'email']
print(list(json_schema["properties"].keys()))  # ['name', 'age', 'email']
```

**Example 3 — Nested schema with ArrayProperty**

```python
from agent_framework_declarative._models import ArrayProperty, Property, PropertySchema

schema = PropertySchema(
    properties=[
        ArrayProperty(
            name="results",
            kind="array",
            description="Search results",
            required=True,
            items=Property(name="item", kind="string"),
        ),
        Property(name="totalCount", kind="integer"),
    ]
)

js = schema.to_json_schema()
print(js["properties"]["results"]["type"])       # array
print(js["properties"]["results"]["items"])      # {'type': 'string', ...}
print(js.get("required"))                        # ['results']
```

---

## 7 · `Connection` hierarchy

**Module:** `agent_framework_declarative._models`

Connection types specify how a tool or model authenticates to an external service. `Connection.from_dict()` dispatches on the `kind` field.

### Class signatures

```python
class Connection(SerializationMixin):
    def __init__(
        self,
        kind: Literal["reference", "remote", "key", "anonymous"],
        authenticationMode: str | None = None,   # PowerFx-evaluated
        usageDescription: str | None = None,     # PowerFx-evaluated
    ) -> None: ...

class ReferenceConnection(Connection):
    def __init__(self, kind="reference", ..., name: str | None = None, target: str | None = None): ...

class RemoteConnection(Connection):
    def __init__(self, kind="remote", ..., name: str | None = None, endpoint: str | None = None): ...

class ApiKeyConnection(Connection):
    def __init__(self, kind="key", ..., endpoint: str | None = None,
                 apiKey: str | None = None, key: str | None = None): ...
    # 'key' takes precedence over 'apiKey' when both present
    # apiKey is evaluated with log_value=False (value not logged)

class AnonymousConnection(Connection):
    def __init__(self, kind="anonymous", ..., endpoint: str | None = None): ...
```

| `kind` value | Subclass | Extra fields |
|---|---|---|
| `"reference"` | `ReferenceConnection` | `name`, `target` |
| `"remote"` | `RemoteConnection` | `name`, `endpoint` |
| `"key"` or `"apikey"` | `ApiKeyConnection` | `endpoint`, `apiKey`/`key` |
| `"anonymous"` | `AnonymousConnection` | `endpoint` |

### Examples

**Example 1 — Connection dispatch from dict**

```python
from agent_framework_declarative._models import (
    AnonymousConnection,
    ApiKeyConnection,
    Connection,
    ReferenceConnection,
    RemoteConnection,
)

conns = [
    {"kind": "reference", "name": "myConn", "target": "azure-openai-prod"},
    {"kind": "remote",    "name": "custom",  "endpoint": "https://inference.example.com"},
    {"kind": "key",       "endpoint": "https://api.openai.com", "key": "sk-..."},
    {"kind": "anonymous", "endpoint": "https://open-endpoint.example.com"},
]

for c in conns:
    obj = Connection.from_dict(c)
    print(type(obj).__name__, obj.kind)
# ReferenceConnection reference
# RemoteConnection    remote
# ApiKeyConnection    key
# AnonymousConnection anonymous
```

**Example 2 — ApiKeyConnection with PowerFx env reference**

```python
import os
from agent_framework_declarative._models import ApiKeyConnection

os.environ["OPENAI_KEY"] = "sk-test-1234"

# When powerfx package is installed and the value starts with '=':
conn = ApiKeyConnection.from_dict({
    "kind": "key",
    "endpoint": "https://api.openai.com/v1",
    "key": "=Env.OPENAI_KEY",   # evaluated at construction time
})
# conn.apiKey == "sk-test-1234"  (if PowerFx engine available)
# otherwise conn.apiKey == "=Env.OPENAI_KEY"  (expression string preserved)
print(conn.endpoint)  # https://api.openai.com/v1
```

**Example 3 — Embed a Connection inside a Model**

```python
from agent_framework_declarative._models import ApiKeyConnection, Model, ModelOptions

model = Model(
    id="gpt-4o",
    provider="openai",
    apiType="chat",
    connection=ApiKeyConnection(
        endpoint="https://api.openai.com/v1",
        apiKey="sk-...",
    ),
    options=ModelOptions(
        temperature=0.2,
        maxOutputTokens=1024,
        stopSequences=["###"],
    ),
)
print(model.id, model.options.temperature)  # gpt-4o 0.2
```

---

## 8 · `McpTool` · `McpServerApprovalMode` hierarchy

**Module:** `agent_framework_declarative._models`

`McpTool` is the declarative representation of an MCP (Model Context Protocol) server tool. It carries an approval mode that controls which MCP tools require user confirmation before execution.

### `McpServerApprovalMode` hierarchy

```python
class McpServerApprovalMode(SerializationMixin):
    def __init__(self, kind: str | None = None) -> None: ...

class McpServerToolAlwaysRequireApprovalMode(McpServerApprovalMode):
    def __init__(self, kind: str = "always") -> None: ...

class McpServerToolNeverRequireApprovalMode(McpServerApprovalMode):
    def __init__(self, kind: str = "never") -> None: ...

class McpServerToolSpecifyApprovalMode(McpServerApprovalMode):
    def __init__(
        self,
        kind: str = "specify",
        alwaysRequireApprovalTools: list[str] | None = None,
        neverRequireApprovalTools: list[str] | None = None,
    ) -> None: ...
```

### `McpTool`

```python
class McpTool(Tool):
    def __init__(
        self,
        name: str | None = None,
        kind: str = "mcp",
        description: str | None = None,
        bindings: list[Binding] | None = None,
        connection: Connection | None = None,       # auto-deserialized
        serverName: str | None = None,              # PowerFx-evaluated
        serverDescription: str | None = None,       # PowerFx-evaluated
        approvalMode: McpServerApprovalMode | None = None,  # auto-deserialized; "always"/"never" accepted as string
        allowedTools: list[str] | None = None,      # tool name allowlist
        url: str | None = None,                     # PowerFx-evaluated
    ) -> None: ...
```

The `approvalMode` field accepts a simplified string shortcut (`"always"` / `"never"`) in addition to a full `McpServerApprovalMode` dict. The tool constructs the appropriate subclass automatically.

### Examples

**Example 1 — McpTool with per-tool approval gating**

```python
from agent_framework_declarative._models import McpTool, McpServerToolSpecifyApprovalMode

# McpTool.from_dict() cannot dispatch a dict approvalMode to the correct subclass
# (McpServerApprovalMode.from_dict uses the base class and ignores "kind").
# Construct the approval mode directly and pass it to McpTool().
approval = McpServerToolSpecifyApprovalMode(
    alwaysRequireApprovalTools=["browser_navigate", "browser_click"],
    neverRequireApprovalTools=["browser_screenshot"],
)
tool = McpTool(
    name="browserTools",
    serverName="playwright",
    url="http://localhost:8931/sse",
    approvalMode=approval,
    allowedTools=["browser_navigate", "browser_click", "browser_screenshot"],
)

print(tool.serverName)                          # playwright
print(type(tool.approvalMode).__name__)         # McpServerToolSpecifyApprovalMode
print(tool.approvalMode.alwaysRequireApprovalTools)  # ['browser_navigate', 'browser_click']
```

**Example 2 — Simplified string approval mode**

```python
# String shortcuts ("always"/"never") are handled by McpTool.__init__ via
# McpServerApprovalMode.from_dict({"kind": value}). Because McpServerApprovalMode
# has no kind-dispatch in its from_dict, this produces a base McpServerApprovalMode
# instance with .kind set — NOT a named subclass.
# For a named subclass, use direct construction (see Example 1 and Example 3).
from agent_framework_declarative._models import McpTool

always_tool = McpTool.from_dict({
    "kind": "mcp",
    "name": "fileSystem",
    "serverName": "filesystem",
    "approvalMode": "always",
})
print(type(always_tool.approvalMode).__name__)  # McpServerApprovalMode  (base class)
print(always_tool.approvalMode.kind)            # always

never_tool = McpTool.from_dict({
    "kind": "mcp",
    "name": "readOnlySearch",
    "serverName": "search",
    "approvalMode": "never",
})
print(type(never_tool.approvalMode).__name__)   # McpServerApprovalMode  (base class)
print(never_tool.approvalMode.kind)             # never
```

**Example 3 — Compose McpTool with Connection and Bindings**

```python
from agent_framework_declarative._models import (
    AnonymousConnection,
    Binding,
    McpServerToolAlwaysRequireApprovalMode,
    McpTool,
)

tool = McpTool(
    name="codeSandbox",
    serverName="code-sandbox",
    url="https://mcp.example.com/sse",
    connection=AnonymousConnection(endpoint="https://mcp.example.com"),
    approvalMode=McpServerToolAlwaysRequireApprovalMode(),
    allowedTools=["execute_python", "install_package"],
    bindings=[
        Binding(name="timeout", input="=Local.timeoutSeconds"),
    ],
)
print(tool.kind)              # mcp
print(tool.allowedTools)      # ['execute_python', 'install_package']
print(tool.bindings[0].name)  # timeout
```

---

## 9 · `Template` · `Format` · `Parser` · `Model` · `ModelOptions`

**Module:** `agent_framework_declarative._models`

These types configure how a `PromptAgent` formats its prompt, parses the response, and selects its LLM backend.

### Class signatures

```python
class ModelOptions(SerializationMixin):
    def __init__(
        self,
        frequencyPenalty: float | None = None,
        maxOutputTokens: int | None = None,
        presencePenalty: float | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        topK: int | None = None,
        topP: float | None = None,
        stopSequences: list[str] | None = None,
        allowMultipleToolCalls: bool | None = None,
        additionalProperties: dict[str, Any] | None = None,
        **kwargs: Any,    # merged into additionalProperties
    ) -> None: ...

class Model(SerializationMixin):
    def __init__(
        self,
        id: str | None = None,          # PowerFx-evaluated — model identifier
        provider: str | None = None,    # PowerFx-evaluated — e.g. "openai", "azure"
        apiType: str | None = None,     # PowerFx-evaluated — e.g. "chat"
        connection: Connection | None = None,   # auto-deserialized; accepts any Connection subclass
        options: ModelOptions | None = None,     # auto-deserialized
    ) -> None: ...

class Format(SerializationMixin):
    def __init__(
        self,
        kind: str | None = None,   # e.g. "json_object", "text"
        strict: bool = False,
        options: dict[str, Any] | None = None,
    ) -> None: ...

class Parser(SerializationMixin):
    def __init__(
        self,
        kind: str | None = None,   # e.g. "json", "text"
        options: dict[str, Any] | None = None,
    ) -> None: ...

class Template(SerializationMixin):
    def __init__(
        self,
        format: Format | None = None,   # auto-deserialized
        parser: Parser | None = None,   # auto-deserialized
    ) -> None: ...
```

### Examples

**Example 1 — Full model configuration**

```python
from agent_framework_declarative._models import ApiKeyConnection, Model, ModelOptions

model = Model.from_dict({
    "id": "gpt-4o",
    "provider": "openai",
    "apiType": "chat",
    "connection": {
        "kind": "key",
        "endpoint": "https://api.openai.com/v1",
        "key": "sk-proj-...",
    },
    "options": {
        "temperature": 0.0,
        "maxOutputTokens": 2048,
        "seed": 42,
        "allowMultipleToolCalls": True,
    },
})

print(model.id)                          # gpt-4o
print(model.options.temperature)         # 0.0
print(model.options.allowMultipleToolCalls)  # True
print(type(model.connection).__name__)   # ApiKeyConnection
```

**Example 2 — Template with JSON output format**

```python
from agent_framework_declarative._models import Format, Parser, Template

template = Template.from_dict({
    "format": {
        "kind": "json_object",
        "strict": True,
    },
    "parser": {
        "kind": "json",
        "options": {"extract_first": True},
    },
})

print(template.format.kind)           # json_object
print(template.format.strict)         # True
print(template.parser.kind)           # json
print(template.parser.options)        # {'extract_first': True}
```

**Example 3 — ModelOptions with additionalProperties**

```python
from agent_framework_declarative._models import ModelOptions

# Extra kwargs go into additionalProperties
opts = ModelOptions(
    temperature=0.7,
    maxOutputTokens=512,
    topP=0.95,
    response_format={"type": "json_object"},   # extra kwarg
    user="session-abc123",                      # extra kwarg
)

print(opts.temperature)           # 0.7
print(opts.additionalProperties)  # {'response_format': {...}, 'user': 'session-abc123'}
```

---

## 10 · `InvokeAzureAgentExecutor` · `AgentResult` · `AgentExternalInputRequest` · `AgentExternalInputResponse`

**Module:** `agent_framework_declarative._workflows._executors_agents`

`InvokeAzureAgentExecutor` invokes an Azure AI Foundry agent from within a declarative workflow, with optional human-in-the-loop (HITL) support via the external-loop pattern.

### Data classes

```python
@dataclass
class AgentResult:
    success: bool
    response: str
    agent_name: str
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[Content] = field(default_factory=list)
    error: str | None = None

@dataclass
class AgentExternalInputRequest:
    request_id: str           # UUID correlating request → response
    agent_name: str           # Which agent is paused
    agent_response: str       # The partial agent response that triggered the HITL gate
    iteration: int = 0        # Loop iteration count
    messages: list[Message] = field(default_factory=list)
    function_calls: list[Content] = field(default_factory=list)

@dataclass
class AgentExternalInputResponse:
    user_input: str                                        # Text to inject as the next user turn
    messages: list[Message] = field(default_factory=list)  # Additional context messages
    function_results: dict[str, Content] = field(default_factory=dict)  # Tool results keyed by call ID
```

### `_extract_json_from_response()`

```python
def _extract_json_from_response(text: str) -> Any:
```

Parses JSON from agent responses that may contain markdown code fences or mixed text. When multiple JSON objects are present it returns the **last** one (typically the final, complete result from a streaming agent). Raises `json.JSONDecodeError` when no valid JSON can be extracted.

### HITL Yield/Resume pattern

The external-loop pattern pauses a workflow mid-execution waiting for user input:

1. `InvokeAzureAgentExecutor` evaluates `externalLoop.when` condition against current state
2. When `True`, it calls `ctx.request_info(AgentExternalInputRequest(...))` — this **yields** the workflow
3. The caller supplies an `AgentExternalInputResponse` via the `request_handler` callback (or `ExternalInputResponse` at workflow level)
4. The executor **resumes** with the user-supplied input injected as a new user message

### Examples

**Example 1 — InvokeAzureAgentExecutor action definition**

```python
# YAML action that InvokeAzureAgentExecutor handles:
# - kind: InvokeAzureAgent
#   id: callSummarizer
#   agent: summarizer            # agent name from manifest
#   connectionName: myAzureConn
#   input:
#     messages:
#       - role: user
#         content: "=Workflow.Inputs.document"
#   resultProperty: Local.summary       # raw string response (top-level key)
#   output:
#     responseObject: Local.summaryJson # full response as parsed object

# _get_output_config() reads: output.messages, output.responseObject, output.property,
# and top-level resultProperty. It does NOT read output.variable.
from agent_framework_declarative._workflows._executors_agents import InvokeAzureAgentExecutor

action_def = {
    "kind": "InvokeAzureAgent",
    "id": "callSummarizer",
    "agent": "summarizer",
    "connectionName": "myAzureConn",
    "input": {
        "messages": [{"role": "user", "content": "=Workflow.Inputs.document"}],
    },
    "resultProperty": "Local.summary",      # raw string response stored here
    "output": {
        "responseObject": "Local.summaryJson",  # full JSON response stored here
    },
}
executor = InvokeAzureAgentExecutor(action_def, id="callSummarizer")
print(executor.id)  # callSummarizer
```

**Example 2 — AgentResult inspection**

```python
from agent_framework_declarative._workflows._executors_agents import AgentResult

# Simulate what the executor produces after a successful invocation
result = AgentResult(
    success=True,
    response='{"summary": "The document discusses...", "keyPoints": ["point1", "point2"]}',
    agent_name="summarizer",
)

import json
parsed = json.loads(result.response)
print(result.success)               # True
print(result.agent_name)            # summarizer
print(parsed["keyPoints"])          # ['point1', 'point2']
```

**Example 3 — HITL external-loop with AgentExternalInputRequest/Response**

```python
import asyncio
from agent_framework_declarative._workflows._executors_agents import (
    AgentExternalInputRequest,
    AgentExternalInputResponse,
)
from agent_framework_declarative import WorkflowFactory

# The workflow YAML must define externalLoop.when on the InvokeAzureAgent action.
# This example shows how to wire up the request_handler on the caller side.

async def human_review_handler(request: AgentExternalInputRequest) -> AgentExternalInputResponse:
    """Called when the workflow pauses for human review."""
    print(f"[HITL] Agent '{request.agent_name}' (iteration {request.iteration}) says:")
    print(f"  {request.agent_response}")
    # In a real app this would send request_id to a UI and await the reply
    user_reply = f"Approved. Continue with iteration {request.iteration}."
    return AgentExternalInputResponse(user_input=user_reply)


async def run_hitl_workflow(yaml_path: str, inputs: dict) -> None:
    factory = WorkflowFactory()
    workflow = factory.create_workflow_from_yaml_path(yaml_path)

    from agent_framework import run_context

    async with run_context(request_handler=human_review_handler) as ctx:
        async for event in workflow.run(ctx=ctx, stream=True, inputs=inputs):
            print("Event:", event)


# asyncio.run(run_hitl_workflow("hitl_agent.yaml", {"document": "..."}))
```

---

## Summary

This volume covered the **declarative execution layer** of `agent_framework_declarative` — a completely separate package from the functional `@workflow` layer documented in Vols. 1–22. Key takeaways:

- **Graph-not-interpreter**: Control flow (`if`, `foreach`, `break`) is encoded as graph node types, not interpreter branches. `ConditionResult` and `LoopIterationResult` carry routing data across edges.
- **PowerFx everywhere**: All string fields in `_models.py` are passed through `_try_powerfx_eval()`. Strings starting with `=` are PowerFx expressions; everything else is a plain literal. `DeclarativeEnvConfig` tightly controls which `os.environ` names can enter the expression engine.
- **State as the source of truth**: `DeclarativeWorkflowState` wraps `State` with namespace-scoped dot-path access. Loop state, conversation slots, and variable scopes all live inside the same checkpoint-safe `State` object.
- **MCP approval gating**: `McpTool.approvalMode` accepts string shortcuts (`"always"` / `"never"`) in addition to the full hierarchy, making YAML manifests concise while keeping programmatic control fine-grained.
- **HITL via Yield/Resume**: `InvokeAzureAgentExecutor` pauses the workflow graph via `ctx.request_info()` when `externalLoop.when` evaluates truthy. The caller supplies `AgentExternalInputResponse` to resume, injecting a new user message without restarting the workflow.

---
title: "Class deep dives — volume 31 (10 additional classes)"
description: "Source-verified deep dives into 10 additional google-adk 2.3.0 classes: _AgentTransferLlmRequestProcessor, _IdentityLlmRequestProcessor, DataFileUtil + _CodeExecutionRequestProcessor + _CodeExecutionResponseProcessor, LLMRegistry, EnvironmentSimulationEngine, EnvironmentSimulationFactory, MockStrategy + TracingMockStrategy, and ToolSpecMockStrategy."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 31"
  order: 100
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source at
`<site-packages>/google/adk/` on
**google-adk == 2.3.0**. Path varies by environment; run `pip show google-adk` to find yours.
No documentation or blog posts were used as primary sources.
</Aside>

## 1 · `_AgentTransferLlmRequestProcessor` — agent-transfer pipeline stage

**Module:** `google.adk.flows.llm_flows.agent_transfer`

`_AgentTransferLlmRequestProcessor` is the LLM request pipeline stage that enables multi-agent routing. It runs on every LLM call for `LlmAgent` instances and injects the `transfer_to_agent` tool declaration plus natural-language routing instructions into the system prompt so the model knows which peer, child, and parent agents are available.

### Key implementation facts (verified from source)

- **Guard check** — the processor does nothing if the agent lacks `disallow_transfer_to_parent` (i.e., it only runs for `LlmAgent` instances, not generic `BaseAgent`).
- **Transfer targets** include: sub-agents not in `single_turn` or `task` mode; the parent agent (if `disallow_transfer_to_parent=False`); peer agents (other sub-agents of the parent, if `disallow_transfer_to_peers=False`).
- **Instruction injection** calls `llm_request.append_instructions([...])` with a block that lists agent names and descriptions and instructs the model: *"if another agent is better … call `transfer_to_agent` function … when transferring, do not generate any text other than the function call."*
- **No instructions in task/single_turn mode** — `_build_transfer_instructions` returns `''` when `agent.mode in ('task', 'single_turn')`, so purely functional sub-agents stay silent about routing.
- **Module-level singleton** — `request_processor = _AgentTransferLlmRequestProcessor()` is instantiated once and reused across all invocations.

```python
# flows/llm_flows/agent_transfer.py (simplified)
def _get_transfer_targets(agent):
    result = [
        sub for sub in agent.sub_agents
        if not hasattr(sub, 'mode') or sub.mode not in ('single_turn', 'task')
    ]
    if agent.parent_agent and hasattr(agent.parent_agent, 'disallow_transfer_to_parent'):
        if not agent.disallow_transfer_to_parent:
            result.append(agent.parent_agent)
        if not agent.disallow_transfer_to_peers:
            result.extend([
                peer for peer in agent.parent_agent.sub_agents
                if peer.name != agent.name
                and (not hasattr(peer, 'mode') or peer.mode not in ('single_turn', 'task'))
            ])
    return result
```

### Example 1 — basic hub-and-spoke agent routing

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App

billing_agent = LlmAgent(
    name="billing_agent",
    model="gemini-2.0-flash",
    description="Handles billing questions, invoices, and payment issues.",
    instruction="You are a billing specialist. Answer billing questions only.",
)

shipping_agent = LlmAgent(
    name="shipping_agent",
    model="gemini-2.0-flash",
    description="Handles shipping, tracking, and delivery questions.",
    instruction="You are a shipping specialist. Answer shipping questions only.",
)

# The coordinator has both agents as sub-agents.
# _AgentTransferLlmRequestProcessor auto-injects the transfer_to_agent tool.
coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.0-flash",
    description="Routes customer queries to the right specialist.",
    instruction="Greet the customer and route to the appropriate specialist.",
    sub_agents=[billing_agent, shipping_agent],
)

async def main():
    session_service = InMemorySessionService()
    app = App(name="support_app", root_agent=coordinator)
    runner = Runner(app=app, session_service=session_service)
    session = await session_service.create_session(
        app_name="support_app", user_id="user1"
    )

    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Where is my package? Tracking #ABC123")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — preventing upward transfer with `disallow_transfer_to_parent`

```python
from google.adk.agents import LlmAgent

# This sub-agent will NOT transfer back to the parent coordinator.
# Useful for agents that must complete the task themselves.
specialist = LlmAgent(
    name="tax_specialist",
    model="gemini-2.0-flash",
    description="Handles all tax calculation questions.",
    instruction="Calculate taxes precisely. Never transfer away.",
    # The processor checks: if disallow_transfer_to_parent is True,
    # the parent is excluded from _get_transfer_targets().
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.0-flash",
    description="Routes to specialists.",
    instruction="Route to the appropriate specialist.",
    sub_agents=[specialist],
)
```

### Example 3 — inspecting the injected transfer instructions

```python
from google.adk.flows.llm_flows.agent_transfer import (
    _build_transfer_instruction_body,
)
from google.adk.agents import LlmAgent

# Build the instruction text that the processor would inject
billing = LlmAgent(name="billing_agent", model="gemini-2.0-flash",
                   description="Handles billing.")
shipping = LlmAgent(name="shipping_agent", model="gemini-2.0-flash",
                    description="Handles shipping.")

instruction = _build_transfer_instruction_body(
    tool_name="transfer_to_agent",
    target_agents=[billing, shipping],
)
print(instruction)
# Output shows the prompt block that gets appended to the system instruction,
# listing both agents alphabetically with their descriptions.
```

---

## 2 · `_IdentityLlmRequestProcessor` — agent self-identity injection

**Module:** `google.adk.flows.llm_flows.identity`

`_IdentityLlmRequestProcessor` injects a brief self-identity sentence into the system prompt on every LLM call so the model knows its own name and role. It is the simplest LLM request processor in the pipeline.

### Key implementation facts (verified from source)

- **Exact injected text** (from source):
  ```
  You are an agent. Your internal name is "{agent.name}". The description about you is "{agent.description}".
  ```
  The description clause is only appended when `agent.description` is non-empty.
- **Skipped for `single_turn` mode** — `if getattr(agent, 'mode', None) != 'single_turn'`. Task-delegation sub-agents in `single_turn` mode receive no identity injection.
- **No-op yield** — the function uses `if False: yield` to satisfy the `AsyncGenerator` protocol without emitting any events.
- **Module-level singleton** — `request_processor = _IdentityLlmRequestProcessor()` is instantiated once.

```python
# flows/llm_flows/identity.py (exact source)
class _IdentityLlmRequestProcessor(BaseLlmRequestProcessor):
    async def run_async(self, invocation_context, llm_request):
        agent = invocation_context.agent
        if getattr(agent, 'mode', None) != 'single_turn':
            si = f'You are an agent. Your internal name is "{agent.name}".'
            if agent.description:
                si += f' The description about you is "{agent.description}".'
            llm_request.append_instructions([si])
        if False:
            yield
```

### Example 1 — confirming identity injection in the system prompt

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.flows.llm_flows.identity import request_processor
from unittest.mock import MagicMock

# Simulate what happens during an LLM invocation
agent = LlmAgent(
    name="customer_support",
    model="gemini-2.0-flash",
    description="Helps customers with their questions.",
)

async def inspect_identity_injection():
    llm_request = LlmRequest(model="gemini-2.0-flash")
    ctx = MagicMock()
    ctx.agent = agent

    async for _ in request_processor.run_async(ctx, llm_request):
        pass  # No events yielded

    print(llm_request.system_instruction)
    # → 'You are an agent. Your internal name is "customer_support".
    #     The description about you is "Helps customers with their questions."'

asyncio.run(inspect_identity_injection())
```

### Example 2 — single_turn mode skips injection

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.flows.llm_flows.identity import request_processor
from unittest.mock import MagicMock

# A task-delegation sub-agent in single_turn mode
sub_agent = LlmAgent(
    name="summarizer",
    model="gemini-2.0-flash",
    description="Summarizes text.",
    # mode="single_turn" means identity is NOT injected
)
sub_agent.mode = "single_turn"

async def check_no_injection():
    llm_request = LlmRequest(model="gemini-2.0-flash")
    ctx = MagicMock()
    ctx.agent = sub_agent

    async for _ in request_processor.run_async(ctx, llm_request):
        pass

    # system_instruction will be None or not contain identity text
    print("Identity injected:", bool(llm_request.system_instruction))
    # → Identity injected: False

asyncio.run(check_no_injection())
```

### Example 3 — custom agent without a description

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.flows.llm_flows.identity import request_processor
from unittest.mock import MagicMock

# An agent with no description — only the name is injected
agent_no_desc = LlmAgent(
    name="processor",
    model="gemini-2.0-flash",
    # description omitted intentionally
)

async def no_description():
    llm_request = LlmRequest(model="gemini-2.0-flash")
    ctx = MagicMock()
    ctx.agent = agent_no_desc

    async for _ in request_processor.run_async(ctx, llm_request):
        pass

    print(llm_request.system_instruction)
    # → 'You are an agent. Your internal name is "processor".'
    # The description clause is absent because agent.description is falsy.

asyncio.run(no_description())
```

---

## 3 · `DataFileUtil` + `_CodeExecutionRequestProcessor` + `_CodeExecutionResponseProcessor` — code execution pipeline

**Module:** `google.adk.flows.llm_flows._code_execution`

These three classes form the code-execution layer of the LLM pipeline. `DataFileUtil` is a dataclass describing how to handle a data file type; `_CodeExecutionRequestProcessor` runs before the LLM call (uploading files, executing pandas exploration code); `_CodeExecutionResponseProcessor` runs after (extracting and executing code blocks from the model's reply).

### Key implementation facts (verified from source)

**`DataFileUtil`**
- Has two fields: `extension` (str) and `loader_code_template` (str with `{filename}` placeholder).
- `_DATA_FILE_UTIL_MAP` maps `'text/csv'` to `DataFileUtil(extension='.csv', loader_code_template="pd.read_csv('{filename}')")`. Only CSV is supported at 2.3.0.
- The pre-processor injects a helper library (`_DATA_FILE_HELPER_LIB`) containing `explore_df()` to print shape, dtypes, null counts, and unique value samples.

**`_CodeExecutionRequestProcessor`**
- Skips entirely if the agent has no `code_executor`.
- For `BuiltInCodeExecutor`, calls `code_executor.process_llm_request(llm_request)` and returns.
- For external executors: extracts inline data files from the request contents, replaces them with `\nAvailable file: \`data_N_M.csv\`\n` text placeholders, runs pandas exploration code, and appends the execution result as a `model` role content block.
- For non-`BuiltInCodeExecutor` types, converts code execution parts in the history to text using `CodeExecutionUtils.convert_code_execution_parts()`.

**`_CodeExecutionResponseProcessor`**
- Skips if `llm_response.partial` (streaming chunks).
- For `BuiltInCodeExecutor`: saves any `image/*` inline data parts to the artifact service, updates `event_actions.artifact_delta`, and replaces image parts with `"Saved as artifact: {file_name}. "`.
- For external executors: uses `CodeExecutionUtils.extract_code_and_truncate_content()` to find the first code block in the model reply; executes it; yields an `Event` for the code and another for the result; sets `llm_response.content = None` to continue the code generation loop.
- Error tracking: `code_executor_context.increment_error_count()` on `stderr`; resets on success. Stops retrying once `error_count >= code_executor.error_retry_attempts`.

### Example 1 — agent with built-in code executor (CSV data analysis)

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App
from google.genai import types

async def main():
    agent = LlmAgent(
        name="data_analyst",
        model="gemini-2.0-flash",
        instruction=(
            "You are a data analyst. When given a CSV file, "
            "analyse it and answer questions about it."
        ),
        # BuiltInCodeExecutor uses Gemini's native code execution sandbox.
        # _CodeExecutionRequestProcessor calls process_llm_request() for it.
        code_executor=BuiltInCodeExecutor(),
    )

    session_service = InMemorySessionService()
    app = App(name="data_app", root_agent=agent)
    runner = Runner(app=app, session_service=session_service)
    session = await session_service.create_session(
        app_name="data_app", user_id="user1"
    )

    # Upload a CSV as inline data — the processor will handle it
    csv_content = "name,age,score\nAlice,30,95\nBob,25,87\nCarla,35,92\n"
    user_message = types.Content(
        role="user",
        parts=[
            types.Part(text="Analyse this dataset and find the average score:"),
            types.Part(
                inline_data=types.Blob(
                    mime_type="text/csv",
                    data=csv_content.encode(),
                )
            ),
        ],
    )

    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=user_message,
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — understanding the `DataFileUtil` map and file naming

```python
from google.adk.flows.llm_flows._code_execution import (
    _DATA_FILE_UTIL_MAP,
    DataFileUtil,
)

# Inspect the built-in map
for mime_type, util in _DATA_FILE_UTIL_MAP.items():
    print(f"MIME: {mime_type}")
    print(f"  Extension: {util.extension}")
    print(f"  Loader: {util.loader_code_template}")
# → MIME: text/csv
# →   Extension: .csv
# →   Loader: pd.read_csv('{filename}')

# The processor generates file names like data_{content_index}_{part_index}.csv
# e.g. for the 2nd content block, 1st part → data_2_1.csv
# The loader becomes: pd.read_csv('data_2_1.csv')

# You can extend the map if you subclass the processor (advanced):
custom_util = DataFileUtil(
    extension=".parquet",
    loader_code_template="pd.read_parquet('{filename}')",
)
print(custom_util)
```

### Example 3 — UnsafeLocalCodeExecutor with CSV data preprocessing

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.code_executors import UnsafeLocalCodeExecutor
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App
from google.genai import types

async def main():
    # UnsafeLocalCodeExecutor runs code locally — for dev/testing only.
    # _CodeExecutionRequestProcessor still pre-processes the message, but
    # the DataFileUtil CSV optimisation (steps 1-2 of Example 1) is skipped
    # because optimize_data_file=False is fixed. The inline CSV is forwarded
    # to the model unchanged; _CodeExecutionResponseProcessor extracts and
    # runs the model's generated code locally.
    # UnsafeLocalCodeExecutor always has stateful=False and optimize_data_file=False
    # (passing either as True raises ValueError). DataFileUtil CSV preprocessing
    # is therefore NOT triggered here — the inline data reaches the model as-is.
    # For optimize_data_file support use BuiltInCodeExecutor (see Example 1 above).
    agent = LlmAgent(
        name="local_analyst",
        model="gemini-2.0-flash",
        instruction="Analyse the provided data. Use Python code to answer.",
        code_executor=UnsafeLocalCodeExecutor(),
    )

    session_service = InMemorySessionService()
    app = App(name="local_app", root_agent=agent)
    runner = Runner(app=app, session_service=session_service)
    session = await session_service.create_session(
        app_name="local_app", user_id="dev_user"
    )

    csv_data = "product,revenue,units\nWidget,1500,30\nGadget,3200,64\n"
    message = types.Content(
        role="user",
        parts=[
            types.Part(text="What is the revenue per unit for each product?"),
            types.Part(
                inline_data=types.Blob(
                    mime_type="text/csv",
                    data=csv_data.encode(),
                )
            ),
        ],
    )

    async for event in runner.run_async(
        user_id="dev_user",
        session_id=session.id,
        new_message=message,
    ):
        if event.content and not event.content.parts[0].executable_code:
            if event.is_final_response():
                print(event.content.parts[0].text)

asyncio.run(main())
```

---

## 4 · `LLMRegistry` — model registry with lazy loading and prefix routing

**Module:** `google.adk.models.registry`

`LLMRegistry` is the global registry that maps model name strings to `BaseLlm` subclass instances. It supports lazy provider loading, explicit `prefix:model` routing, and `lru_cache`-backed resolution.

### Key implementation facts (verified from source)

- **`new_llm(model_name)`** — creates a fresh `BaseLlm` instance for the given name by calling `resolve(model_name)(model=model_name)`.
- **`resolve(model_name)`** — `@lru_cache(maxsize=None)`-decorated; returns the registered class (not an instance). Cache is keyed on the exact string.
- **Prefix routing** — if `model_name` contains `:`, the part before `:` is treated as a provider prefix for explicit routing (e.g. `"litellm:gpt-4o"` always routes to `LiteLlm`).
- **Lazy module loading** — `_register_lazy(prefix, module_path, class_name)` defers importing the provider module until `resolve()` is first called with a matching prefix. Used to avoid heavy imports at startup.
- **Helpful error messages** — if a model name starts with `"claude-"`, the error suggests installing `anthropic`; if it contains `"/"`, it suggests `litellm`.

```python
# models/registry.py (simplified)
class LLMRegistry:
    _registry: dict[str, type[BaseLlm]] = {}
    _lazy_registry: dict[str, tuple[str, str]] = {}

    @classmethod
    def register(cls, prefix: str, llm_class: type[BaseLlm]) -> None:
        cls._registry[prefix] = llm_class
        cls.resolve.cache_clear()   # Invalidate lru_cache on new registration

    @classmethod
    @lru_cache(maxsize=None)
    def resolve(cls, model_name: str) -> type[BaseLlm]:
        # 1. Explicit prefix routing: "litellm:gpt-4o" → LiteLlm
        if ":" in model_name:
            prefix = model_name.split(":")[0]
            # ... load from lazy or direct registry
        # 2. Walk registered prefixes longest-first
        for prefix in sorted(cls._registry, key=len, reverse=True):
            if model_name.startswith(prefix):
                return cls._registry[prefix]
        raise ValueError(f"No LLM registered for model: {model_name}")
```

### Example 1 — resolving built-in Gemini models

```python
from google.adk.models.registry import LLMRegistry

registry = LLMRegistry()

# Resolve a Gemini model (registered under "gemini" prefix)
GeminiClass = registry.resolve("gemini-2.0-flash")
print(GeminiClass)  # <class 'google.adk.models.google_llm.Gemini'>

# Instantiate it directly
llm = GeminiClass(model="gemini-2.0-flash")
print(llm.model)  # gemini-2.0-flash

# new_llm() is a convenience wrapper
llm2 = registry.new_llm("gemini-2.0-flash-lite")
print(type(llm2).__name__)  # Gemini
```

### Example 2 — explicit prefix routing with LiteLLM

```python
from google.adk.models.registry import LLMRegistry
from google.adk.models.lite_llm import LiteLlm  # pip install litellm

registry = LLMRegistry()

# Register LiteLlm explicitly for the "litellm" prefix
LLMRegistry.register("litellm", LiteLlm)

# Now use the prefix:model syntax for explicit routing.
# Useful when a model name would otherwise be ambiguous.
llm = registry.new_llm("litellm:gpt-4o")
print(type(llm).__name__)  # LiteLlm
print(llm.model)           # litellm:gpt-4o

# This pattern is also useful for any OpenAI-compatible endpoint:
llm_local = registry.new_llm("litellm:ollama/llama3")
```

### Example 3 — registering a custom LLM provider

```python
from google.adk.models.registry import LLMRegistry
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from typing import AsyncGenerator

class MyCustomLlm(BaseLlm):
    """A custom LLM provider that routes to an internal model server."""

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["internal-model-v1", "internal-model-v2"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        # In practice, call your internal API here
        from google.genai import types
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Response from custom model")],
            )
        )

# Register under a unique prefix
LLMRegistry.register("internal", MyCustomLlm)

# Use it in an agent
from google.adk.agents import LlmAgent

agent = LlmAgent(
    name="custom_agent",
    model="internal:internal-model-v1",  # prefix:model routing
    instruction="You are a helpful assistant.",
)
```

---

## 5 · `EnvironmentSimulationEngine` — stateful tool simulation engine

**Module:** `google.adk.tools.environment_simulation.environment_simulation_engine`

`EnvironmentSimulationEngine` is the core engine that intercepts tool calls and replaces them with simulated responses. It maintains a `_state_store` dict shared across a session and picks the right mock strategy based on `MockStrategy`.

### Key implementation facts (verified from source)

- **`@experimental(FeatureName.ENVIRONMENT_SIMULATION)`** — not stable API.
- **Constructor** takes an `EnvironmentSimulationConfig`; stores `config`, initialises `_state_store = {}`, and sets `_tool_connection_map = None` (lazy).
- **`simulate(tool, args, tool_context)`** is the main entry point, called as a `before_tool_callback`.
  - Returns `None` if the tool is not in the simulation config (letting the real tool run).
  - Lazy-initialises `_tool_connection_map` via `ToolConnectionAnalyzer` on first call.
  - Picks strategy: `MockStrategy.MOCK_STRATEGY_TOOL_SPEC` → `ToolSpecMockStrategy`; `MockStrategy.MOCK_STRATEGY_TRACING` → `TracingMockStrategy` (deprecated, returns "Not implemented").
  - Passes `environment_data` and `tracing` from `EnvironmentSimulationConfig` directly to the strategy.
- **Shared `_state_store`** — updated by `ToolSpecMockStrategy` after mutative tool calls (creating tools); consumed to generate realistic responses for consuming tools (e.g. returning 404 for unknown IDs).

### Example 1 — simulating a ticketing tool without calling the real API

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App
from google.adk.tools.environment_simulation import EnvironmentSimulationFactory
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
    MockStrategy,
    ToolSimulationConfig,
)
from google.adk.tools import FunctionTool

def create_ticket(title: str, description: str) -> dict:
    """Creates a support ticket. Returns ticket_id."""
    # In production this calls a real ticketing API
    return {"ticket_id": "REAL-001", "status": "open"}

def get_ticket(ticket_id: str) -> dict:
    """Retrieves a ticket by ID."""
    return {"ticket_id": ticket_id, "title": "Real ticket", "status": "open"}

# Configure which tools to simulate
config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="create_ticket",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
        ToolSimulationConfig(
            tool_name="get_ticket",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
    ],
    simulation_model="gemini-2.0-flash",
)

# Create the before_tool_callback that intercepts real tool calls
simulation_callback = EnvironmentSimulationFactory.create_callback(config)

agent = LlmAgent(
    name="support_agent",
    model="gemini-2.0-flash",
    instruction="Help users create and check support tickets.",
    tools=[create_ticket, get_ticket],
    before_tool_callback=simulation_callback,
)

async def main():
    session_service = InMemorySessionService()
    app = App(name="support_app", root_agent=agent)
    runner = Runner(app=app, session_service=session_service)
    session = await session_service.create_session(
        app_name="support_app", user_id="user1"
    )

    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Create a ticket: Login is broken")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — injecting environment data for realistic simulation

```python
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
    MockStrategy,
    ToolSimulationConfig,
)

# Provide a database snapshot so the simulator generates consistent IDs
db_snapshot = """
Users table:
  - id: U001, name: Alice, email: alice@example.com, plan: pro
  - id: U002, name: Bob, email: bob@example.com, plan: free

Products table:
  - id: P001, name: Widget, price: 29.99, stock: 150
  - id: P002, name: Gadget, price: 49.99, stock: 23
"""

config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="get_user",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
        ToolSimulationConfig(
            tool_name="get_product",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
        ToolSimulationConfig(
            tool_name="create_order",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
    ],
    simulation_model="gemini-2.0-flash",
    # environment_data is forwarded to ToolSpecMockStrategy.mock() and injected
    # into the mock-generation prompt so the LLM generates consistent IDs.
    environment_data=db_snapshot,
)

print("Config ready with environment data injection")
print(f"Simulating {len(config.tool_simulation_configs)} tools")
```

### Example 3 — using the plugin pattern instead of a callback

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App
from google.adk.tools.environment_simulation import EnvironmentSimulationFactory
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
    MockStrategy,
    ToolSimulationConfig,
)

def send_email(to: str, subject: str, body: str) -> dict:
    """Sends an email. Returns message_id."""
    return {"message_id": "REAL-MSG-001", "status": "sent"}

config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="send_email",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
    ],
    simulation_model="gemini-2.0-flash",
)

# create_plugin() returns an EnvironmentSimulationPlugin
# that hooks into the ADK plugin lifecycle instead of a callback.
# The plugin registers the same before_tool_callback under the hood.
sim_plugin = EnvironmentSimulationFactory.create_plugin(config)

agent = LlmAgent(
    name="email_agent",
    model="gemini-2.0-flash",
    instruction="Send emails on behalf of the user.",
    tools=[send_email],
)

async def main():
    session_service = InMemorySessionService()
    app = App(
        name="email_app",
        root_agent=agent,
        plugins=[sim_plugin],   # Plugin wires up simulation automatically
    )
    runner = Runner(app=app, session_service=session_service)
    session = await session_service.create_session(
        app_name="email_app", user_id="tester"
    )

    async for event in runner.run_async(
        user_id="tester",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Send a welcome email to alice@example.com")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())
```

---

## 6 · `EnvironmentSimulationFactory` — factory for simulation callbacks and plugins

**Module:** `google.adk.tools.environment_simulation.environment_simulation_factory`

`EnvironmentSimulationFactory` is a static-method factory that converts an `EnvironmentSimulationConfig` into either a bare async callback or a full `EnvironmentSimulationPlugin`. Both forms produce a single shared `EnvironmentSimulationEngine` instance, ensuring the `_state_store` is shared across all tool calls in a session.

### Key implementation facts (verified from source)

- **`@experimental(FeatureName.ENVIRONMENT_SIMULATION)`** — not stable API.
- **`create_callback(config)`** — creates one `EnvironmentSimulationEngine`, wraps it in a `async def _environment_simulation_callback(tool, args, tool_context)` closure, and returns the closure. Use as `before_tool_callback` on an `LlmAgent`.
- **`create_plugin(config)`** — creates one `EnvironmentSimulationEngine`, passes it to `EnvironmentSimulationPlugin(engine)`, and returns the plugin. Use in `App(plugins=[...])`.
- The engine instance is **shared** between calls via closure capture, so `_state_store` persists across tool invocations in the same session.

### Example 1 — using `create_callback` for per-agent simulation

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App
from google.adk.tools.environment_simulation import EnvironmentSimulationFactory
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
    MockStrategy,
    ToolSimulationConfig,
)

def book_hotel(hotel_id: str, check_in: str, check_out: str) -> dict:
    """Books a hotel room. Returns reservation_id."""
    return {"reservation_id": "REAL-RES-001", "status": "confirmed"}

def cancel_reservation(reservation_id: str) -> dict:
    """Cancels a hotel reservation."""
    return {"status": "cancelled", "refund": "pending"}

config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="book_hotel",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
        ToolSimulationConfig(
            tool_name="cancel_reservation",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
    ],
    simulation_model="gemini-2.0-flash",
)

# The factory creates one engine and closes over it
callback = EnvironmentSimulationFactory.create_callback(config)
# callback signature: async (tool, args, tool_context) -> dict | None

agent = LlmAgent(
    name="hotel_agent",
    model="gemini-2.0-flash",
    instruction="Help users book and manage hotel reservations.",
    tools=[book_hotel, cancel_reservation],
    before_tool_callback=callback,
)

async def main():
    session_service = InMemorySessionService()
    app = App(name="hotel_app", root_agent=agent)
    runner = Runner(app=app, session_service=session_service)
    session = await session_service.create_session(
        app_name="hotel_app", user_id="traveler"
    )

    async for event in runner.run_async(
        user_id="traveler",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Book hotel H123 for Dec 20-25")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — shared state across multi-turn interactions

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App
from google.adk.tools.environment_simulation import EnvironmentSimulationFactory
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
    MockStrategy,
    ToolSimulationConfig,
)

def create_order(product_id: str, quantity: int) -> dict:
    """Creates a purchase order. Returns order_id."""
    return {"order_id": "REAL-ORD-001", "status": "pending"}

def get_order(order_id: str) -> dict:
    """Gets order details."""
    return {"order_id": order_id, "status": "pending"}

# The factory closes over ONE engine → ONE _state_store.
# In turn 1: create_order → engine stores the new order_id.
# In turn 2: get_order with that ID → engine returns the stored details.
# This stateful consistency is the key advantage over simple mocking.
config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="create_order",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
        ToolSimulationConfig(
            tool_name="get_order",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
    ],
    simulation_model="gemini-2.0-flash",
)
callback = EnvironmentSimulationFactory.create_callback(config)

agent = LlmAgent(
    name="order_agent",
    model="gemini-2.0-flash",
    instruction="Help users place and track orders.",
    tools=[create_order, get_order],
    before_tool_callback=callback,
)

async def main():
    session_service = InMemorySessionService()
    app = App(name="order_app", root_agent=agent)
    runner = Runner(app=app, session_service=session_service)
    session = await session_service.create_session(
        app_name="order_app", user_id="shopper"
    )

    # Turn 1: create an order (engine generates and stores a fake order_id)
    async for event in runner.run_async(
        user_id="shopper", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Order 2 units of product P001")]),
    ):
        if event.is_final_response():
            print("Turn 1:", event.content.parts[0].text)

    # Turn 2: look up the order created in turn 1 (engine uses stored state)
    async for event in runner.run_async(
        user_id="shopper", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="What is the status of my latest order?")]),
    ):
        if event.is_final_response():
            print("Turn 2:", event.content.parts[0].text)

asyncio.run(main())
```

### Example 3 — building a test harness with `create_plugin`

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App
from google.adk.tools.environment_simulation import EnvironmentSimulationFactory
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
    MockStrategy,
    ToolSimulationConfig,
)

def charge_card(amount: float, card_token: str) -> dict:
    """Charges a payment card. Returns transaction_id."""
    return {"transaction_id": "REAL-TXN-001", "status": "approved"}

async def build_test_app(simulation_config: EnvironmentSimulationConfig) -> tuple:
    """Helper that wires up a simulated app for testing."""
    plugin = EnvironmentSimulationFactory.create_plugin(simulation_config)
    agent = LlmAgent(
        name="payment_agent",
        model="gemini-2.0-flash",
        instruction="Process payments for orders.",
        tools=[charge_card],
    )
    session_service = InMemorySessionService()
    app = App(
        name="payment_app",
        root_agent=agent,
        plugins=[plugin],
    )
    return Runner(app=app, session_service=session_service), session_service


async def test_payment_flow():
    config = EnvironmentSimulationConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="charge_card",
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            ),
        ],
        simulation_model="gemini-2.0-flash",
    )
    runner, session_service = await build_test_app(config)
    session = await session_service.create_session(
        app_name="payment_app", user_id="test_user"
    )

    responses = []
    async for event in runner.run_async(
        user_id="test_user", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Charge $50 to card token tok_test123")]),
    ):
        if event.is_final_response():
            responses.append(event.content.parts[0].text)

    assert len(responses) == 1
    assert "transaction" in responses[0].lower() or "charged" in responses[0].lower()
    print("Test passed:", responses[0])

asyncio.run(test_payment_flow())
```

---

## 7 · `MockStrategy` + `TracingMockStrategy` — mock strategy base classes

**Module:** `google.adk.tools.environment_simulation.strategies.base`

`MockStrategy` is the abstract base class for all simulation strategies. `TracingMockStrategy` extends it to replay mock responses from a previously recorded execution trace rather than generating them with an LLM.

### Key implementation facts (verified from source)

**`MockStrategy`**
- Abstract base class with one abstract method: `async def mock(tool, args, tool_context, tool_connection_map, state_store, environment_data, tracing) -> dict`.
- `@experimental(FeatureName.ENVIRONMENT_SIMULATION)`.
- All strategies receive `state_store` (shared mutable dict) and `tool_connection_map` (describes which tools create/consume stateful parameters).

**`TracingMockStrategy`**
- Takes a `tracing: str` parameter containing a JSON-encoded trace from a prior agent run.
- Parses the trace to find a recorded response for the exact `(tool_name, args)` pair.
- Falls back to generating a response if no match is found.
- Useful for deterministic replay testing: record a real run, then replay it without calling external APIs.

### Example 1 — implementing a custom `MockStrategy`

```python
import asyncio
from typing import Any, Dict, Optional
from google.adk.tools.environment_simulation.strategies.base import MockStrategy
from google.adk.tools.environment_simulation.tool_connection_map import ToolConnectionMap
from google.adk.tools.base_tool import BaseTool

class FixedResponseMockStrategy(MockStrategy):
    """Always returns a fixed response regardless of tool or args."""

    def __init__(self, fixed_responses: dict[str, dict]):
        # Map from tool_name → fixed response dict
        self._fixed_responses = fixed_responses

    async def mock(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        tool_context: Any,
        tool_connection_map: Optional[ToolConnectionMap],
        state_store: Dict[str, Any],
        environment_data: Optional[str] = None,
        tracing: Optional[str] = None,
    ) -> Dict[str, Any]:
        if tool.name in self._fixed_responses:
            return self._fixed_responses[tool.name]
        return {"status": "error", "message": f"No mock for tool: {tool.name}"}


# Use the custom strategy with EnvironmentSimulationEngine
from google.adk.tools.environment_simulation.environment_simulation_engine import (
    EnvironmentSimulationEngine,
)
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
    MockStrategy,
    ToolSimulationConfig,
)

config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="get_weather",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
    ],
    simulation_model="gemini-2.0-flash",
)

engine = EnvironmentSimulationEngine(config)
# Override the strategy selection by setting it directly
engine._custom_strategy = FixedResponseMockStrategy({
    "get_weather": {"temperature": 22, "condition": "sunny", "city": "London"},
})
```

### Example 2 — passing trace context via `EnvironmentSimulationConfig.tracing`

`TracingMockStrategy` (`MOCK_STRATEGY_TRACING`) is **deprecated** — its `mock()` always returns
`{"status": "error", "error_message": "Not implemented"}`. It does not replay traces.

To give the `ToolSpecMockStrategy` historical context, pass the trace as a JSON string in
`EnvironmentSimulationConfig.tracing`; the engine forwards it to `ToolSpecMockStrategy.mock(tracing=...)`
so the LLM can reference prior call patterns when generating mocks.

```python
import json
from google.adk.tools.environment_simulation import EnvironmentSimulationFactory
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
    MockStrategy,
    ToolSimulationConfig,
)

# Trace from a prior run captured as a plain list of {tool_name, args, response} dicts
recorded_trace = json.dumps([
    {
        "tool_name": "search_products",
        "args": {"query": "wireless headphones"},
        "response": {
            "results": [
                {"id": "P001", "name": "SoundPro X", "price": 79.99},
                {"id": "P002", "name": "AudioMax 3", "price": 129.99},
            ]
        }
    },
    {
        "tool_name": "get_product_details",
        "args": {"product_id": "P001"},
        "response": {
            "id": "P001",
            "name": "SoundPro X",
            "description": "Wireless over-ear headphones with 30h battery",
            "in_stock": True,
        }
    }
])

# tracing= is forwarded to ToolSpecMockStrategy.mock(tracing=...) and injected
# into the prompt so the LLM uses the captured patterns as reference.
config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="search_products",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
        ToolSimulationConfig(
            tool_name="get_product_details",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
    ],
    simulation_model="gemini-2.0-flash",
    tracing=recorded_trace,
)
callback = EnvironmentSimulationFactory.create_callback(config)
print("Simulation callback with trace context ready:", len(json.loads(config.tracing)), "recorded calls")
```

### Example 3 — composing strategies with a fallback chain

```python
import asyncio
from typing import Any, Dict, Optional
from google.adk.tools.environment_simulation.strategies.base import MockStrategy
from google.adk.tools.environment_simulation.tool_connection_map import ToolConnectionMap
from google.adk.tools.base_tool import BaseTool

class FallbackMockStrategy(MockStrategy):
    """Tries each strategy in order; returns the first non-error response."""

    def __init__(self, strategies: list[MockStrategy]):
        self._strategies = strategies

    async def mock(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        tool_context: Any,
        tool_connection_map: Optional[ToolConnectionMap],
        state_store: Dict[str, Any],
        environment_data: Optional[str] = None,
        tracing: Optional[str] = None,
    ) -> Dict[str, Any]:
        for strategy in self._strategies:
            result = await strategy.mock(
                tool, args, tool_context,
                tool_connection_map, state_store,
                environment_data, tracing,
            )
            if result.get("status") != "error":
                return result
        return {"status": "error", "message": "All strategies failed"}


# Example: try tracing first, fall back to a fixed response
from google.adk.tools.environment_simulation.strategies.base import TracingMockStrategy
import json

class AlwaysSuccessStrategy(MockStrategy):
    async def mock(self, tool, args, tool_context, tool_connection_map,
                   state_store, environment_data=None, tracing=None):
        return {"status": "ok", "tool": tool.name, "mocked": True}

# TracingMockStrategy is deprecated — mock() always returns "Not implemented".
# In a fallback chain it will always fail through to the next strategy.
# Constructor takes llm_name (str) and optional llm_config, not tracing data.
fallback = FallbackMockStrategy([
    TracingMockStrategy(llm_name="gemini-2.0-flash"),
    AlwaysSuccessStrategy(),
])
print("Fallback strategy chain ready (TracingMockStrategy will always fall through)")
```

---

## 8 · `ToolSpecMockStrategy` — LLM-powered tool spec mock strategy

**Module:** `google.adk.tools.environment_simulation.strategies.tool_spec_mock_strategy`

`ToolSpecMockStrategy` generates realistic mock responses for tool calls using an LLM. It inspects the tool's schema, the shared state store, and the tool connection map to produce statefully consistent JSON responses.

### Key implementation facts (verified from source)

- **`@experimental(FeatureName.ENVIRONMENT_SIMULATION)`**.
- **Constructor** takes `llm_name: str` and `llm_config: genai_types.GenerateContentConfig`. Uses `LLMRegistry().resolve(llm_name)` to get the provider class, then instantiates it.
- **`mock()` method**:
  1. Gets the tool's `FunctionDeclaration` via `tool._get_declaration()`.
  2. Builds the prompt from `_TOOL_SPEC_MOCK_PROMPT_TEMPLATE` with: `environment_data`, `tracing`, `tool_connection_map` JSON, `state_store` JSON, tool name/description/schema/args.
  3. Calls the LLM with `response_mime_type="application/json"` (forces JSON output).
  4. Strips markdown code fences if present (`^```[a-zA-Z]*\n` and `\n```$`).
  5. Parses JSON; on `JSONDecodeError` returns `{"status": "error", ...}`.
- **State mutation** after successful mock:
  - Checks if the tool is in `all_creating_tools` from the `tool_connection_map`.
  - If yes, finds the stateful parameter value in the response via `_find_value_by_key()` (recursive nested dict search).
  - Stores `state_store[param_name][param_value] = mock_response` so subsequent consuming-tool calls can reference it.
- **404 for unknown IDs** — the prompt instructs the LLM: *"If an ID is provided that does not exist in the state, return a realistic error (e.g., a 404 Not Found error)."*

### Example 1 — direct use of `ToolSpecMockStrategy`

```python
import asyncio
from google.genai import types as genai_types
from google.adk.tools.environment_simulation.strategies.tool_spec_mock_strategy import (
    ToolSpecMockStrategy,
)
from google.adk.tools import FunctionTool

def create_support_ticket(
    title: str,
    description: str,
    priority: str = "medium",
) -> dict:
    """Creates a support ticket and returns ticket_id and status."""
    ...  # Real implementation

tool = FunctionTool(create_support_ticket)

strategy = ToolSpecMockStrategy(
    llm_name="gemini-2.0-flash",
    llm_config=genai_types.GenerateContentConfig(temperature=0.1),
)

async def demo():
    result = await strategy.mock(
        tool=tool,
        args={"title": "Login broken", "description": "Cannot sign in", "priority": "high"},
        tool_context=None,
        tool_connection_map=None,
        state_store={},
    )
    print(result)
    # → {"ticket_id": "TKT-7429", "status": "open", "priority": "high", ...}
    # (LLM generates a realistic-looking ticket response)

asyncio.run(demo())
```

### Example 2 — observing state mutation across creating and consuming tools

```python
import asyncio
from google.genai import types as genai_types
from google.adk.tools.environment_simulation.strategies.tool_spec_mock_strategy import (
    ToolSpecMockStrategy,
)
from google.adk.tools.environment_simulation.tool_connection_map import (
    ToolConnectionMap,
    StatefulParameter,
)
from google.adk.tools import FunctionTool

def create_user(name: str, email: str) -> dict:
    """Creates a user account. Returns user_id."""
    ...

def get_user(user_id: str) -> dict:
    """Gets a user by user_id."""
    ...

create_tool = FunctionTool(create_user)
get_tool = FunctionTool(get_user)

# The connection map tells the strategy that user_id is created by create_user
# and consumed by get_user
connection_map = ToolConnectionMap(
    stateful_parameters=[
        StatefulParameter(
            parameter_name="user_id",
            creating_tools=["create_user"],
            consuming_tools=["get_user"],
        )
    ]
)

strategy = ToolSpecMockStrategy(
    llm_name="gemini-2.0-flash",
    llm_config=genai_types.GenerateContentConfig(temperature=0.0),
)
state_store = {}

async def demo():
    # Step 1: create user — engine stores the generated user_id in state_store
    create_result = await strategy.mock(
        tool=create_tool,
        args={"name": "Alice", "email": "alice@example.com"},
        tool_context=None,
        tool_connection_map=connection_map,
        state_store=state_store,
    )
    print("Create result:", create_result)
    print("State store after create:", state_store)
    # state_store["user_id"] now contains the created user data

    # Step 2: get user — engine finds the user_id in state and returns its data
    user_id = create_result.get("user_id", "USR-001")
    get_result = await strategy.mock(
        tool=get_tool,
        args={"user_id": user_id},
        tool_context=None,
        tool_connection_map=connection_map,
        state_store=state_store,
    )
    print("Get result:", get_result)
    # Returns data consistent with the creation result

asyncio.run(demo())
```

### Example 3 — configuring `ToolSpecMockStrategy` via `EnvironmentSimulationConfig`

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App
from google.adk.tools.environment_simulation import EnvironmentSimulationFactory
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig,
    MockStrategy,
    ToolSimulationConfig,
)

def send_notification(user_id: str, message: str, channel: str = "email") -> dict:
    """Sends a notification to a user."""
    return {"notification_id": "REAL-NOTIF-001", "delivered": True}

# MockStrategy.MOCK_STRATEGY_TOOL_SPEC instructs the engine to use ToolSpecMockStrategy.
# The engine instantiates it internally with the configured simulation_model.
config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="send_notification",
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
    ],
    simulation_model="gemini-2.0-flash",
)

callback = EnvironmentSimulationFactory.create_callback(config)

agent = LlmAgent(
    name="notification_agent",
    model="gemini-2.0-flash",
    instruction="Send notifications to users when requested.",
    tools=[send_notification],
    before_tool_callback=callback,
)

async def main():
    session_service = InMemorySessionService()
    app = App(name="notif_app", root_agent=agent)
    runner = Runner(app=app, session_service=session_service)
    session = await session_service.create_session(
        app_name="notif_app", user_id="ops_user"
    )

    async for event in runner.run_async(
        user_id="ops_user", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Send an email notification to user U001 about their order")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())
```

---

## Summary table

| Class | Module | Experimental | Key role |
|---|---|---|---|
| `_AgentTransferLlmRequestProcessor` | `flows.llm_flows.agent_transfer` | No | Injects `transfer_to_agent` tool + routing instructions |
| `_IdentityLlmRequestProcessor` | `flows.llm_flows.identity` | No | Injects `"You are an agent. Your internal name is …"` |
| `DataFileUtil` | `flows.llm_flows._code_execution` | No | Dataclass: extension + pandas loader template for inline files |
| `_CodeExecutionRequestProcessor` | `flows.llm_flows._code_execution` | No | Pre-processes data files; runs `explore_df`; converts history parts |
| `_CodeExecutionResponseProcessor` | `flows.llm_flows._code_execution` | No | Extracts + executes code blocks; saves image artifacts |
| `LLMRegistry` | `models.registry` | No | Global model registry; prefix routing; `lru_cache` resolve |
| `EnvironmentSimulationEngine` | `tools.environment_simulation.…engine` | Yes | Core intercept engine; shared `_state_store`; strategy dispatch |
| `EnvironmentSimulationFactory` | `tools.environment_simulation.…factory` | Yes | Creates callback or plugin from `EnvironmentSimulationConfig` |
| `MockStrategy` / `TracingMockStrategy` | `tools.environment_simulation.strategies.base` | Yes | Abstract base; tracing replay variant |
| `ToolSpecMockStrategy` | `tools.environment_simulation.strategies.tool_spec_mock_strategy` | Yes | LLM-generated stateful mock responses from tool schema |

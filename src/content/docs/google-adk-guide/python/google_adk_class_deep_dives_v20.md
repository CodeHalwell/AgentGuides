---
title: "Class deep dives — volume 20 (2.3.0)"
description: "Source-verified 2.3.0 deep dives: McpToolset (all 4 connection types, auth, resources, sampling), BuiltInCodeExecutor (file I/O, stateful, data file optimisation), UnsafeLocalCodeExecutor (subprocess sandbox, timeout), PlanReActPlanner (5-tag system, replanning), BuiltInPlanner (ThinkingConfig, budget_tokens), RunConfig 2.3.0 (translation, affective dialog, proactivity, session resumption, context window compression), InvocationContext (max_llm_calls, canonical_tools_cache, event queue), Event + NodeInfo (output field, is_final_response, message kwarg), App + EventsCompactionConfig + ResumabilityConfig (production setup), Context unified API (state, artifacts, memory, auth, confirmation)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 20"
  order: 89
---

Source-verified against **google-adk==2.3.0** (installed from PyPI, June 2026). Every field name, signature, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `McpToolset` + connection params | `google.adk.tools.mcp_tool.mcp_toolset` | Stable |
| 2 | `BuiltInCodeExecutor` | `google.adk.code_executors` | Stable |
| 3 | `UnsafeLocalCodeExecutor` | `google.adk.code_executors` | Stable |
| 4 | `PlanReActPlanner` | `google.adk.planners` | Stable |
| 5 | `BuiltInPlanner` | `google.adk.planners` | Stable |
| 6 | `RunConfig` + `StreamingMode` 2.3.0 | `google.adk.agents.run_config` | Stable |
| 7 | `InvocationContext` 2.3.0 | `google.adk.agents.invocation_context` | Stable |
| 8 | `Event` + `NodeInfo` | `google.adk.events.event` | Stable |
| 9 | `App` + `EventsCompactionConfig` + `ResumabilityConfig` | `google.adk.apps` | Stable / `@experimental` |
| 10 | `Context` (unified CallbackContext / ToolContext) | `google.adk.agents.context` | Stable |

---

## 1 · `McpToolset` — all four connection types, auth, resources, and sampling

**Source:** `google.adk.tools.mcp_tool.mcp_toolset`

`McpToolset` bridges any MCP-compatible server into ADK as a `BaseToolset`. It wraps
an `MCPSessionManager` that handles connection pooling, reconnection, and auth. Pass an
instance directly inside `tools=[...]` on any `LlmAgent`.

### Constructor (source-verified, 2.3.0)

```python
McpToolset(
    *,
    connection_params: Union[
        StdioServerParameters,       # legacy; no timeout support
        StdioConnectionParams,       # preferred stdio
        SseConnectionParams,         # SSE HTTP
        StreamableHTTPConnectionParams,  # new in 2.3.0: Streamable HTTP
    ],
    tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
    tool_name_prefix: Optional[str] = None,
    errlog: TextIO = sys.stderr,
    auth_scheme: Optional[AuthScheme] = None,
    auth_credential: Optional[AuthCredential] = None,
    require_confirmation: Union[bool, Callable[..., bool]] = False,
    header_provider: Optional[Callable[[ReadonlyContext], Dict[str, str]]] = None,
    progress_callback: Optional[Union[ProgressFnT, ProgressCallbackFactory]] = None,
    use_mcp_resources: Optional[bool] = False,
    sampling_callback: Optional[SamplingFnT] = None,
    sampling_capabilities: Optional[SamplingCapability] = None,
    credential_key: str | None = None,
)
```

### Connection parameter types

```python
from mcp import StdioServerParameters
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StdioConnectionParams,
    SseConnectionParams,
    StreamableHTTPConnectionParams,
)

# 1. Stdio (preferred): wrap an npx/python3 subprocess, with configurable timeout
stdio_params = StdioConnectionParams(
    server_params=StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    ),
    timeout=10.0,   # seconds to establish the connection
)

# 2. SSE HTTP: point at a remote or local SSE endpoint
sse_params = SseConnectionParams(
    url="http://localhost:3000/sse",
    headers={"X-Api-Key": "secret"},
    timeout=5.0,
    sse_read_timeout=300.0,   # 5-minute read timeout for long operations
)

# 3. Streamable HTTP (new in 2.3.0): modern MCP over HTTP/2
streamable_params = StreamableHTTPConnectionParams(
    url="https://mcp.example.com/mcp",
    headers={"Authorization": "Bearer <token>"},
    timeout=5.0,
    sse_read_timeout=300.0,
    terminate_on_close=True,
)
```

### Example 1 — filesystem tools via stdio

```python
import asyncio
from mcp import StdioServerParameters
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams

async def main():
    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            ),
            timeout=10.0,
        ),
        # Only expose read operations
        tool_filter=["read_file", "list_directory", "get_file_info"],
        tool_name_prefix="fs_",   # tools become fs_read_file, fs_list_directory …
    )

    agent = LlmAgent(
        name="file_agent",
        model="gemini-2.5-flash",
        instruction="Help users read and navigate the filesystem.",
        tools=[toolset],
    )

    runner = InMemoryRunner(agent=agent, app_name="fs-demo")
    await runner.session_service.create_session(
        app_name="fs-demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "List everything in /tmp", user_id="u1", session_id="s1"
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — tool filter predicate

Use a `ToolPredicate` callable when you need dynamic filtering logic:

```python
from google.adk.tools.base_toolset import ToolPredicate
from google.adk.agents.readonly_context import ReadonlyContext

def only_read_tools(tool: object, readonly_context: ReadonlyContext = None) -> bool:
    return "read" in tool.name.lower() or "list" in tool.name.lower()

toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
    ),
    tool_filter=only_read_tools,
)
```

### Example 3 — Streamable HTTP with dynamic auth headers

When auth tokens need to be generated per-request (e.g. short-lived JWT), use
`header_provider`:

```python
import time, hmac, hashlib
from google.adk.agents.readonly_context import ReadonlyContext

def sign_request(ctx: ReadonlyContext) -> dict[str, str]:
    ts = str(int(time.time()))
    sig = hmac.new(b"shared-secret", ts.encode(), hashlib.sha256).hexdigest()
    return {"X-Timestamp": ts, "X-Signature": sig, "X-User": ctx.user_id}

toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://mcp.example.com/mcp",
        terminate_on_close=True,
    ),
    header_provider=sign_request,
)
```

### Example 4 — MCP resources + progress callbacks

Enable MCP resources to let the agent read resource URIs; attach a progress
callback to surface long-operation progress to session state:

```python
from typing import Optional
from google.adk.tools.mcp_tool.mcp_tool import ProgressCallbackFactory
from google.adk.agents.context import Context

def build_progress_cb(
    tool_name: str,
    *,
    callback_context: Context | None = None,
    **kwargs,
):
    """ProgressCallbackFactory: called once per tool; returns a ProgressFnT."""
    async def on_progress(
        progress: float, total: Optional[float], message: Optional[str]
    ) -> None:
        pct = (progress / total * 100) if total else None
        if callback_context:
            callback_context.state["last_progress"] = {
                "tool": tool_name,
                "progress": progress,
                "total": total,
                "pct": pct,
                "message": message,
            }
    return on_progress

toolset = McpToolset(
    connection_params=SseConnectionParams(url="http://localhost:3000/sse"),
    use_mcp_resources=True,        # adds load_mcp_resource tool; injects resource list
    progress_callback=build_progress_cb,
    require_confirmation=True,     # all tools in this toolset need user confirmation
)
```

### Example 5 — OAuth2 auth for a protected MCP server

```python
from google.adk.auth.auth_schemes import OAuthGrantType, OpenIdConnectWithConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth

toolset = McpToolset(
    connection_params=SseConnectionParams(url="https://api.example.com/mcp"),
    auth_scheme=OpenIdConnectWithConfig(
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        grant_type=OAuthGrantType.CLIENT_CREDENTIALS,
        scopes=["mcp:tools"],
    ),
    auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="my-client",
            client_secret="my-secret",
        ),
    ),
    credential_key="example_mcp_oauth",   # stored/loaded from CredentialService
)
```

---

## 2 · `BuiltInCodeExecutor` — Gemini-native code execution

**Source:** `google.adk.code_executors.built_in_code_executor`

`BuiltInCodeExecutor` delegates Python execution to the Gemini model itself via
`types.ToolCodeExecution()`. Requires Gemini 2.0+ (checked via
`is_gemini_eap_or_2_or_above(model)` at request time). No subprocess is spawned;
the model handles the sandbox.

### Constructor (inherits `BaseCodeExecutor` fields)

```python
from google.adk.code_executors import BuiltInCodeExecutor

executor = BuiltInCodeExecutor(
    optimize_data_file=False,       # extract CSV files and attach to executor
    stateful=False,                 # maintain a persistent kernel between calls
    error_retry_attempts=2,         # retry on consecutive code errors
    code_block_delimiters=[         # delimiters to identify code blocks
        ('```tool_code\n', '\n```'),
        ('```python\n', '\n```'),
    ],
    execution_result_delimiters=('```tool_output\n', '\n```'),
    timeout_seconds=None,           # no timeout (model enforces its own)
)
```

### Attaching to an `LlmAgent`

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor

data_analyst = LlmAgent(
    name="analyst",
    model="gemini-2.5-flash",
    instruction=(
        "You are a data analyst. When given data, write Python code to analyse "
        "it. Show your work. Use pandas and matplotlib."
    ),
    code_executor=BuiltInCodeExecutor(
        optimize_data_file=True,     # parses CSVs from user messages for the model
        error_retry_attempts=3,      # retry up to 3 times on code errors
    ),
)
```

### Example — stateful session for iterative analysis

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.runners import InMemoryRunner
import asyncio

agent = LlmAgent(
    name="notebook",
    model="gemini-2.5-pro",
    instruction="Act as a Python REPL. Maintain state between messages.",
    code_executor=BuiltInCodeExecutor(stateful=True),
)

async def run_notebook():
    runner = InMemoryRunner(agent=agent, app_name="repl")
    await runner.session_service.create_session(
        app_name="repl", user_id="u1", session_id="s1"
    )
    # Turn 1: define a variable
    await runner.run_debug("x = [1, 2, 3, 4, 5]; print(sum(x))",
                           user_id="u1", session_id="s1")
    # Turn 2: x is still defined (stateful=True)
    events = await runner.run_debug("print(x)",
                                    user_id="u1", session_id="s1")
    print(events[-1].content.parts[0].text)

asyncio.run(run_notebook())
```

### `CodeExecutionInput` and `CodeExecutionResult` data types

These dataclasses flow into `BaseCodeExecutor.execute_code()` — useful when
building a custom executor:

```python
from google.adk.code_executors.code_execution_utils import (
    CodeExecutionInput, CodeExecutionResult, File
)

# Input: code + optional files for the executor
inp = CodeExecutionInput(
    code="import csv; print(list(csv.reader(open('data.csv'))))",
    input_files=[
        File(name="data.csv", content=b"a,b\n1,2\n3,4", mime_type="text/csv")
    ],
    execution_id="session-123",   # required when stateful=True
)

# Result: stdout + stderr + output files
result = CodeExecutionResult(
    stdout="[['a', 'b'], ['1', '2'], ['3', '4']]",
    stderr="",
    output_files=[
        File(name="plot.png", content=b"\x89PNG...", mime_type="image/png")
    ],
)
```

### Comparison table

| Feature | `BuiltInCodeExecutor` | `UnsafeLocalCodeExecutor` | `VertexAiCodeExecutor` |
|---|---|---|---|
| Runtime | Gemini model sandbox | Spawned subprocess | Vertex AI managed |
| Model requirement | Gemini 2.0+ | Any | Any |
| `stateful` | ✓ (model maintains kernel) | ✗ (always false) | ✓ |
| `optimize_data_file` | ✓ | ✗ | ✓ |
| Output files | ✓ (model returns) | ✓ | ✓ |
| Timeout control | Model-side | `timeout_seconds` | `timeout_seconds` |
| GCP billing | Included in Gemini call | None | Separate |

---

## 3 · `UnsafeLocalCodeExecutor` — subprocess sandbox

**Source:** `google.adk.code_executors.unsafe_local_code_executor`

`UnsafeLocalCodeExecutor` runs Python code in a **separate spawned process** using
`multiprocessing.get_context('spawn')`. The "Unsafe" prefix means there is no
container or sandbox boundary — the spawned process inherits the same OS user
and file-system access. Use only in trusted, isolated environments.

### Key constraints (from source)

```python
# Both fields are frozen — cannot be overridden at construction time.
stateful: bool = Field(default=False, frozen=True, exclude=True)
optimize_data_file: bool = Field(default=False, frozen=True, exclude=True)
```

If you attempt `UnsafeLocalCodeExecutor(stateful=True)`, a `ValueError` is raised
immediately in `__init__`.

### Constructor

```python
from google.adk.code_executors import UnsafeLocalCodeExecutor

executor = UnsafeLocalCodeExecutor(
    error_retry_attempts=2,
    code_block_delimiters=[
        ('```python\n', '\n```'),
        ('```tool_code\n', '\n```'),
    ],
    timeout_seconds=10,    # kills the subprocess after 10s
)
```

### Execution mechanics

The executor calls `_execute_in_process(code, globals_, result_queue)` in a
`spawn`-mode child process:

1. `exec(code, globals_, globals_)` runs the snippet in an empty global namespace.
2. `stdout` is captured via `contextlib.redirect_stdout`.
3. Any exception is captured via `traceback.format_exc()` into `stderr`.
4. The `(stdout, error)` tuple is sent back via a `multiprocessing.Queue`.
5. If `result_queue.get(timeout=self.timeout_seconds)` raises `queue.Empty`,
   the process is terminated and an error string is returned.

```python
# Patterns supported by _prepare_globals:
# If code checks `if __name__ == "__main__":`, __name__ is set to '__main__'
code = """
if __name__ == '__main__':
    print("Hello from subprocess!")
"""
```

### Attaching to an agent

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors import UnsafeLocalCodeExecutor

calculator = LlmAgent(
    name="calculator",
    model="gemini-2.5-flash",
    instruction=(
        "When asked to calculate, write Python code inside a ```python block "
        "and I will execute it for you."
    ),
    code_executor=UnsafeLocalCodeExecutor(timeout_seconds=5),
)
```

### Custom delimiter example

Override `code_block_delimiters` to match a custom block format:

```python
from google.adk.code_executors import UnsafeLocalCodeExecutor

executor = UnsafeLocalCodeExecutor(
    code_block_delimiters=[
        ('<exec>', '</exec>'),    # custom XML-like block
        ('```python\n', '\n```'), # keep standard markdown too
    ],
    execution_result_delimiters=('<result>', '</result>'),
    error_retry_attempts=1,
    timeout_seconds=30,
)
```

### Example — safe LLM-driven tests

Use `UnsafeLocalCodeExecutor` to let an agent run generated pytest snippets:

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.code_executors import UnsafeLocalCodeExecutor
from google.adk.runners import InMemoryRunner

test_runner_agent = LlmAgent(
    name="test_writer",
    model="gemini-2.5-pro",
    instruction=(
        "Write a pytest function to test the user's requirement. "
        "Wrap the test inside a ```python block. "
        "Import only stdlib modules. "
        "Print PASS or FAIL at the end."
    ),
    code_executor=UnsafeLocalCodeExecutor(
        timeout_seconds=15,
        error_retry_attempts=2,
    ),
)

async def main():
    runner = InMemoryRunner(agent=test_runner_agent, app_name="tests")
    await runner.session_service.create_session(
        app_name="tests", user_id="dev", session_id="s1"
    )
    events = await runner.run_debug(
        "Test that sorted([3,1,2]) == [1,2,3]",
        user_id="dev", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

---

## 4 · `PlanReActPlanner` — explicit Plan → Re-Act loop

**Source:** `google.adk.planners.plan_re_act_planner`

`PlanReActPlanner` injects a structured system instruction that forces the model
to produce responses tagged with one of five sentinel strings before any tool
calls or final answers. Unlike `BuiltInPlanner`, it works on **all** models
including those without native thinking support.

### Tag constants (source-verified)

```python
PLANNING_TAG    = '/*PLANNING*/'
REPLANNING_TAG  = '/*REPLANNING*/'
REASONING_TAG   = '/*REASONING*/'
ACTION_TAG      = '/*ACTION*/'
FINAL_ANSWER_TAG = '/*FINAL_ANSWER*/'
```

### How the response is structured

A typical model response looks like:

```
/*PLANNING*/
1. Search for the GDP of the top 5 countries.
2. Calculate the total.
3. Return the ranked list.

/*ACTION*/
[function call: web_search("GDP top 5 countries 2025")]

/*REASONING*/
The search returned figures in USD trillions. I now have all the data I need.

/*FINAL_ANSWER*/
The top 5 countries by GDP (2025) are: USA $29T, China $19T, Germany $4.6T, Japan $4.2T, India $4.1T.
Total: ~$61 trillion.
```

### How `process_planning_response` processes parts

1. Iterates response parts in order.
2. At the first `function_call` part → collects it (and consecutive FC parts), stops.
3. For text parts: if the text starts with `PLANNING_TAG`, `REASONING_TAG`,
   `ACTION_TAG`, or `REPLANNING_TAG`, marks the part as `thought=True` (hidden
   from the user but kept in context).
4. If text contains `FINAL_ANSWER_TAG`, splits the part: the prefix becomes a
   thought, the suffix becomes visible content.

### Example 1 — attaching to an agent

```python
from google.adk.agents import LlmAgent
from google.adk.planners import PlanReActPlanner
from google.adk.tools import google_search

research_agent = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",    # no thinking required
    instruction="Answer questions using web search.",
    tools=[google_search],
    planner=PlanReActPlanner(),
)
```

### Example 2 — multi-tool research pipeline

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.planners import PlanReActPlanner
from google.adk.tools import FunctionTool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.runners import InMemoryRunner

import ast
import operator as _op

_SAFE_OPS = {
    ast.Add: _op.add, ast.Sub: _op.sub,
    ast.Mult: _op.mul, ast.Div: _op.truediv,
    ast.USub: _op.neg,
}

def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")

def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression (+, -, *, /).

    Args:
      expression: A simple arithmetic expression e.g. '1.2 + 3.4 * 2'.
    """
    return _eval_node(ast.parse(expression, mode='eval').body)

agent = LlmAgent(
    name="analyst",
    model="gemini-2.5-flash",
    instruction="Combine search results with calculations to answer questions.",
    tools=[GoogleSearchTool(bypass_multi_tools_limit=True), FunctionTool(func=calculate)],
    planner=PlanReActPlanner(),
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="plan-demo")
    await runner.session_service.create_session(
        app_name="plan-demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What is 15% of the GDP of Germany in 2024?",
        user_id="u1", session_id="s1",
    )
    for e in events:
        if e.is_final_response():
            print(e.content.parts[0].text)

asyncio.run(main())
```

### Example 3 — inspecting planning thoughts

Thoughts are preserved in session events with `part.thought=True`:

```python
async def run_and_show_thinking(runner, user_id, session_id, query):
    events = await runner.run_debug(query, user_id=user_id, session_id=session_id)
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, 'thought', False) and part.text:
                    print("THOUGHT:", part.text[:200])
                elif part.text and not getattr(part, 'thought', False):
                    print("RESPONSE:", part.text)
```

### When to use `PlanReActPlanner` vs `BuiltInPlanner`

| | `PlanReActPlanner` | `BuiltInPlanner` |
|---|---|---|
| Model requirement | Any model | Gemini 2.0+ with thinking support |
| Overhead | Extra prompt tokens | Native model feature |
| Thought visibility | `part.thought=True` on tagged text | Native `part.thought=True` |
| Replanning | ✓ `/*REPLANNING*/` tag | Implicit (model decides) |
| Latency | +1 turn structure overhead | Minimal |

---

## 5 · `BuiltInPlanner` — model native thinking

**Source:** `google.adk.planners.built_in_planner`

`BuiltInPlanner` enables Gemini's native extended thinking by injecting a
`ThinkingConfig` into every `LlmRequest`. No extra prompt tokens are added;
the model reasons internally and exposes thoughts as `part.thought=True` parts.

### Constructor (source-verified)

```python
from google.adk.planners import BuiltInPlanner
from google.genai import types

planner = BuiltInPlanner(
    thinking_config=types.ThinkingConfig(
        include_thoughts=True,     # surface thought parts in the response
        thinking_budget=8192,      # max tokens for thinking (default varies by model)
    )
)
```

### `apply_thinking_config` mechanics

`BuiltInPlanner.apply_thinking_config(llm_request)` merges the `thinking_config`
into `llm_request.config`, overwriting any existing `thinking_config` on the
request (with a debug log warning).

The methods `build_planning_instruction` and `process_planning_response` both
return `None` — the planner has no prompt-side effect.

### Example 1 — minimal thinking agent

```python
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search
from google.genai import types

thinking_agent = LlmAgent(
    name="thinker",
    model="gemini-2.5-pro",    # must support native thinking
    instruction="Think carefully before answering. Use search when uncertain.",
    tools=[google_search],
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=16384,
        )
    ),
)
```

### Example 2 — reading thought summaries

```python
import asyncio
from google.adk.runners import InMemoryRunner

async def main():
    runner = InMemoryRunner(agent=thinking_agent, app_name="thinking-demo")
    await runner.session_service.create_session(
        app_name="thinking-demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Explain the trolley problem and its main ethical implications.",
        user_id="u1", session_id="s1",
    )
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, 'thought', False) and part.text:
                    print(f"[THOUGHT] {part.text[:300]}")
                elif part.text:
                    print(f"[REPLY] {part.text}")

asyncio.run(main())
```

### Example 3 — `BuiltInPlanner` with disabled thoughts (budget only)

Useful when you want extended thinking for better answers but don't need to
expose internal reasoning to users:

```python
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

silent_thinker = LlmAgent(
    name="silent_thinker",
    model="gemini-2.5-pro",
    instruction="You are a careful assistant. Think before responding.",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,   # thoughts consumed internally; not surfaced
            thinking_budget=4096,
        )
    ),
)
```

### Example 4 — dynamic `thinking_budget` via `RunConfig`

The agent-level `BuiltInPlanner` sets the default; you can override
`thinking_config` per-request via `llm_request.config`:

```python
from google.adk.agents.run_config import RunConfig
from google.genai import types

# High budget for complex queries
complex_config = RunConfig(
    # other run settings …
)
# The planner always overrides thinking_config on the LlmRequest,
# so adjust the planner's thinking_config per agent instance when
# you need different budgets for different agents.
fast_planner = BuiltInPlanner(
    thinking_config=types.ThinkingConfig(thinking_budget=1024)
)
deep_planner = BuiltInPlanner(
    thinking_config=types.ThinkingConfig(thinking_budget=32768)
)
```

---

## 6 · `RunConfig` + `StreamingMode` — complete 2.3.0 field reference

**Source:** `google.adk.agents.run_config`

`RunConfig` controls runtime behaviour for a single `runner.run_async()` call.
Fields not set fall back to agent-level or app-level defaults.

### Complete field table (2.3.0, source-verified)

| Field | Type | Default | Purpose |
|---|---|---|---|
| `streaming_mode` | `StreamingMode` | `NONE` | Streaming behaviour: `NONE`, `SSE`, `BIDI` |
| `max_llm_calls` | `int` | `500` | Hard cap on LLM calls per invocation (≤0 = unlimited) |
| `speech_config` | `types.SpeechConfig\|None` | `None` | TTS voice / config for live agents |
| `response_modalities` | `list[str]\|None` | `None` | Output modalities (default: `AUDIO` in live mode) |
| `avatar_config` | `types.AvatarConfig\|None` | `None` | Avatar config for live agents |
| `support_cfc` | `bool` | `False` | Enable Compositional Function Calling (experimental; SSE only) |
| `output_audio_transcription` | `types.AudioTranscriptionConfig\|None` | factory default | Live audio output transcription |
| `input_audio_transcription` | `types.AudioTranscriptionConfig\|None` | factory default | Live audio input transcription |
| `realtime_input_config` | `types.RealtimeInputConfig\|None` | `None` | Real-time audio input config for live agents |
| `translation_config` | `types.TranslationConfig\|None` | `None` | Speech-to-speech translation (models like `gemini-3.5-live-translate-preview`) |
| `enable_affective_dialog` | `bool\|None` | `None` | Model detects emotions and adapts responses |
| `proactivity` | `types.ProactivityConfig\|None` | `None` | Controls proactive model responses |
| `session_resumption` | `types.SessionResumptionConfig\|None` | `None` | Transparent session resumption mode |
| `history_config` | `types.HistoryConfig\|None` | `None` | Client/server history exchange config |
| `context_window_compression` | `types.ContextWindowCompressionConfig\|None` | `None` | Gemini-side context window compression |
| `save_live_blob` | `bool` | `False` | Saves live audio+video blobs to artifact service |
| `tool_thread_pool_config` | `ToolThreadPoolConfig\|None` | `None` | Thread pool for blocking tool I/O in live mode |
| `telemetry` | `TelemetryConfig\|None` | `None` | Per-request OTEL config (experimental) |
| `get_session_config` | `GetSessionConfig\|None` | `None` | Limit events fetched when loading a session |
| `custom_metadata` | `dict[str,Any]\|None` | `None` | Arbitrary invocation metadata |
| `save_input_blobs_as_artifacts` | `bool` | `False` | **Deprecated** — use `SaveFilesAsArtifactsPlugin` |
| `save_live_audio` | `bool` | `False` | **Deprecated** — use `save_live_blob` |

### Example 1 — SSE streaming with duplicate-text prevention

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = LlmAgent(name="streamer", model="gemini-2.5-flash",
                 instruction="Write concise essays.")

async def stream_response(query: str):
    runner = InMemoryRunner(agent=agent, app_name="sse-demo")
    await runner.session_service.create_session(
        app_name="sse-demo", user_id="u1", session_id="s1"
    )
    config = RunConfig(streaming_mode=StreamingMode.SSE)
    displayed = ""

    async for event in runner.run_async(
        user_id="u1", session_id="s1",
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
        run_config=config,
    ):
        if event.content and event.content.parts:
            text_parts = [p.text or "" for p in event.content.parts if p.text]
            chunk = "".join(text_parts)
            if event.partial and chunk:
                # Each partial event carries an incremental chunk — print and accumulate
                print(chunk, end="", flush=True)
                displayed += chunk
            elif not event.partial and event.is_final_response():
                # Final event: print any text not yet shown, then add newline
                remaining = chunk[len(displayed):]
                if remaining:
                    print(remaining, end="")
                print()  # newline after completion

asyncio.run(stream_response("Write a short essay on entropy."))
```

### Example 2 — `max_llm_calls` guard

```python
from google.adk.agents.run_config import RunConfig
from google.adk.agents.invocation_context import LlmCallsLimitExceededError

config = RunConfig(max_llm_calls=10)

try:
    events = await runner.run_debug(
        "Recursively search and summarise 20 papers on quantum computing.",
        user_id="u1", session_id="s1",
        run_config=config,
    )
except LlmCallsLimitExceededError as exc:
    print(f"Stopped: {exc}")
```

### Example 3 — context window compression

Pass a `ContextWindowCompressionConfig` to enable Gemini-side context pruning
when the prompt grows too long:

```python
from google.adk.agents.run_config import RunConfig
from google.genai import types

config = RunConfig(
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=100_000,       # compress when prompt exceeds 100K tokens
        target_tokens=50_000,         # aim to reduce to 50K
    ),
)
```

### Example 4 — translation live agent

Use `translation_config` for real-time speech-to-speech translation:

```python
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.genai import types

translator = LlmAgent(
    name="translator",
    model="gemini-3.5-live-translate-preview",
    instruction="Translate the user's speech in real time.",
)

live_config = RunConfig(
    response_modalities=["AUDIO"],
    translation_config=types.TranslationConfig(
        target_language_code="es",   # translate to Spanish (BCP-47 code)
    ),
)
```

### Example 5 — session history limiting + `GetSessionConfig`

When sessions are very long, load only the most recent events to reduce
session-service latency:

```python
from google.adk.agents.run_config import RunConfig
from google.adk.sessions.base_session_service import GetSessionConfig

config = RunConfig(
    get_session_config=GetSessionConfig(
        num_recent_events=50,   # only load last 50 events from the session store
    ),
)
```

### Example 6 — `ToolThreadPoolConfig` for blocking tool I/O in live mode

```python
from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig, StreamingMode

config = RunConfig(
    streaming_mode=StreamingMode.BIDI,
    tool_thread_pool_config=ToolThreadPoolConfig(max_workers=8),
    # GIL note: thread pool helps with I/O (DB queries, network) but NOT
    # CPU-bound pure-Python code. Use multiprocessing for CPU work.
)
```

---

## 7 · `InvocationContext` — invocation lifecycle backbone (2.3.0)

**Source:** `google.adk.agents.invocation_context`

An `InvocationContext` is created by the `Runner` at the start of each
`run_async()` call and threaded through every agent, node, callback, and tool
for the lifetime of that invocation. It carries services, the current session,
and cross-cutting state.

### Lifecycle diagram

```
runner.run_async()
└── new InvocationContext(invocation_id=..., session=..., run_config=...)
    └── agent.run(ctx)                  ← one "agent call"
        └── llm_flow.run_step(ctx)     ← one "step" = one LLM call + tool calls
            ├── ctx.increment_llm_call_count()  # enforces max_llm_calls
            ├── tool_1.run_async(ctx)
            └── tool_2.run_async(ctx)
        # repeat until is_final_response or end_invocation or transfer
    └── next_agent.run(ctx)            ← agent transfer reuses same ctx
```

### Key 2.3.0 fields (source-verified)

```python
from google.adk.agents.invocation_context import InvocationContext

# Commonly accessed in callbacks and tools:
ctx.invocation_id          # str  — unique per run_async() call
ctx.session                # Session — current session object
ctx.agent                  # BaseAgent | BaseNode | None — currently running agent
ctx.user_content           # types.Content | None — original user message
ctx.run_config             # RunConfig | None — per-invocation config
ctx.end_invocation         # bool — set True to abort the entire invocation
ctx.branch                 # str | None — "parent.child.grandchild" path

# 2.3.0 fields:
ctx.canonical_tools_cache  # list[BaseTool] | None — cached resolved tools for LLM
ctx.token_compaction_checked  # bool — True once token-threshold compaction ran

# Internal cost tracking (via _InvocationCostManager):
# ctx.increment_llm_call_count() — raises LlmCallsLimitExceededError when exceeded
```

### Example 1 — reading invocation context in a callback

```python
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from typing import Optional
from google.genai import types

def before_model(callback_context: Context, llm_request) -> Optional[types.GenerateContentConfig]:
    """Log every LLM call with its invocation ID."""
    print(f"[{callback_context.invocation_id}] LLM call by {callback_context.agent_name}")
    print(f"  session: {callback_context.session.id}, user: {callback_context.user_id}")
    # Access the underlying InvocationContext:
    inv_ctx = callback_context._invocation_context
    print(f"  end_invocation: {inv_ctx.end_invocation}")
    print(f"  compaction_checked: {inv_ctx.token_compaction_checked}")
    return None   # no change to the request

agent = LlmAgent(
    name="monitored",
    model="gemini-2.5-flash",
    instruction="Answer questions.",
    before_model_callback=before_model,
)
```

### Example 2 — aborting an invocation from a tool

Set `invocation_context.end_invocation = True` from any tool to stop processing
after the current step:

```python
from google.adk.tools.tool_context import ToolContext

def safety_check(user_request: str, tool_context: ToolContext) -> str:
    """Reject requests containing banned keywords."""
    banned = {"hack", "exploit", "bypass"}
    if any(w in user_request.lower() for w in banned):
        tool_context._invocation_context.end_invocation = True
        return "Request rejected: policy violation."
    return f"Request '{user_request}' is safe."
```

### Example 3 — agent state management in a resumable workflow

`set_agent_state` and `reset_sub_agent_states` are used internally by the
workflow engine; understanding them helps when debugging resumable invocations:

```python
from google.adk.agents.base_agent import BaseAgentState

# Mark an agent as having produced some intermediate output:
ctx.set_agent_state("worker_agent", agent_state=BaseAgentState())

# Mark an agent as finished:
ctx.set_agent_state("worker_agent", end_of_agent=True)

# Reset all sub-agents of "orchestrator" (useful when replanning):
ctx.reset_sub_agent_states("orchestrator")
```

### `_InvocationCostManager` and `LlmCallsLimitExceededError`

```python
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner

runner = InMemoryRunner(agent=my_agent, app_name="demo")

try:
    events = await runner.run_debug(
        "Summarise 100 research papers",
        user_id="u1", session_id="s1",
        run_config=RunConfig(max_llm_calls=5),
    )
except LlmCallsLimitExceededError:
    print("Invocation aborted: too many LLM calls.")
```

---

## 8 · `Event` + `NodeInfo` — the streaming event model

**Source:** `google.adk.events.event`

Every piece of information flowing through an ADK invocation — user turns, model
responses, tool calls, tool results, state deltas — is an `Event`. It extends
`LlmResponse` (from `google-genai`) with ADK-specific fields.

### Field table (source-verified, 2.3.0)

| Field | Type | Purpose |
|---|---|---|
| `id` | `str` | Auto-generated UUID (set in `model_post_init`) |
| `invocation_id` | `str` | Links event to its invocation |
| `author` | `str` | `"user"` or agent name |
| `content` | `types.Content\|None` | Text + function calls + code execution parts |
| `actions` | `EventActions` | State delta, route, end-of-agent signals |
| `partial` | `bool` | `True` for SSE streaming chunks |
| `output` | `Any\|None` | Generic output for workflow nodes |
| `node_info` | `NodeInfo` | Workflow path metadata |
| `long_running_tool_ids` | `set[str]\|None` | IDs of HITL/async tool calls |
| `branch` | `str\|None` | `"parent.child"` branch path |
| `isolation_scope` | `str\|None` | Internal scope tag (do not use directly) |
| `timestamp` | `float` | Unix timestamp (auto-set) |

### `NodeInfo` fields

| Field | Type | Purpose |
|---|---|---|
| `path` | `str` | Workflow node path e.g. `"wf/A@1/B@1"` |
| `output_for` | `list[str]\|None` | Node paths this event's `output` applies to |
| `message_as_output` | `bool\|None` | Content is the node's output (no separate output event) |

### Convenience constructor kwargs

`Event` accepts these kwargs and routes them to nested fields:

- `message` → `content` (via `t_content` transformer; mutually exclusive with `content`)
- `state` → `actions.state_delta`
- `route` → `actions.route`
- `node_path` → `node_info.path`

### Example 1 — reading events from `run_debug`

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner

agent = LlmAgent(name="demo", model="gemini-2.5-flash",
                 instruction="Answer concisely.")

async def main():
    runner = InMemoryRunner(agent=agent, app_name="events-demo")
    await runner.session_service.create_session(
        app_name="events-demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug("What is 2 + 2?", user_id="u1", session_id="s1")

    for ev in events:
        print(f"author={ev.author!r}  partial={ev.partial}  "
              f"final={ev.is_final_response()}")
        if ev.content and ev.content.parts:
            for part in ev.content.parts:
                if part.text:
                    print(f"  text: {part.text!r}")
                if part.function_call:
                    print(f"  fc: {part.function_call.name}({part.function_call.args})")

asyncio.run(main())
```

### Example 2 — `is_final_response()` logic (source-verified)

```python
def is_final_response(self) -> bool:
    # True when:
    # 1. actions.skip_summarization is set (end-of-function-response)
    # 2. long_running_tool_ids is non-empty (HITL/async tool call)
    # 3. no function calls, no function responses, not partial,
    #    no trailing code execution result
    ...
```

In practice — filter events like this:

```python
final_events = [ev for ev in events if ev.is_final_response()]
text = final_events[-1].content.parts[0].text
```

### Example 3 — constructing events with `message` kwarg

The `message` convenience kwarg is useful in tests and custom runners:

```python
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions

# Equivalent to passing `content=types.Content(role="user", parts=[...])`:
user_event = Event(
    author="user",
    message="Hello, can you help me?",   # auto-converted to types.Content
    invocation_id="inv-001",
)

# State delta — update session state via an event:
state_event = Event(
    author="my_agent",
    state={"user_name": "Alice", "step": 3},  # → actions.state_delta
    invocation_id="inv-001",
)
print(state_event.actions.state_delta)
# {'user_name': 'Alice', 'step': 3}
```

### Example 4 — SSE streaming event handling pattern

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = LlmAgent(name="streamer", model="gemini-2.5-flash",
                 instruction="Write in detail.")

async def stream_with_events():
    runner = InMemoryRunner(agent=agent, app_name="sse")
    await runner.session_service.create_session(
        app_name="sse", user_id="u1", session_id="s1"
    )
    config = RunConfig(streaming_mode=StreamingMode.SSE)
    msg = types.Content(role="user", parts=[types.Part(text="Explain gravity.")])

    async for event in runner.run_async(
        user_id="u1", session_id="s1",
        new_message=msg,
        run_config=config,
    ):
        if event.partial and event.content and event.content.parts:
            # Partial SSE chunk — could be text or function call arguments
            for part in event.content.parts:
                if part.text and not part.function_call:
                    print(part.text, end="", flush=True)
        else:
            # Final aggregated event
            if event.is_final_response():
                print()  # newline

asyncio.run(stream_with_events())
```

---

## 9 · `App` + `EventsCompactionConfig` + `ResumabilityConfig` — production setup

**Source:** `google.adk.apps.app`, `google.adk.apps._configs`

`App` is the top-level container for an agentic system. It wires together a
root agent (or workflow node), plugins, and optional compaction / resumability /
context-cache configs. `App` instances are consumed by `Runner`.

### `App` constructor

```python
from google.adk.apps.app import App

App(
    name: str,                                    # required; validated by validate_app_name
    root_agent: Union[BaseAgent, BaseNode],        # required; single root
    plugins: list[BasePlugin] = [],               # app-wide plugins
    events_compaction_config: Optional[EventsCompactionConfig] = None,
    context_cache_config: Optional[ContextCacheConfig] = None,
    resumability_config: Optional[ResumabilityConfig] = None,
)
```

`validate_app_name` enforces `^[a-zA-Z][a-zA-Z0-9_-]*$` and rejects the
reserved name `"user"`.

### `EventsCompactionConfig` constructor

```python
from google.adk.apps._configs import EventsCompactionConfig

EventsCompactionConfig(
    compaction_interval: int,              # required; compact every N user invocations
    overlap_size: int,                     # required; include N prior invocations in each compaction
    token_threshold: Optional[int] = None, # post-invocation trigger (must pair with event_retention_size)
    event_retention_size: Optional[int] = None,  # keep last N raw events after token compaction
    summarizer: Optional[BaseEventsSummarizer] = None,  # custom summarizer (default: LlmEventSummarizer)
)
```

`token_threshold` and `event_retention_size` must **both** be set or **both**
be `None` (enforced by `model_validator`).

### `ResumabilityConfig` constructor

```python
from google.adk.apps._configs import ResumabilityConfig

ResumabilityConfig(
    is_resumable: bool = False,   # enables pause/resume on long-running tools
)
```

Requires a persistent `SessionService` (e.g. `DatabaseSessionService` or
`VertexAiSessionService`) — `InMemorySessionService` does not persist state.

### Example 1 — minimal `App` wiring with `Runner`

```python
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

root = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

app = App(name="my-assistant", root_agent=root)
session_svc = InMemorySessionService()
runner = Runner(app=app, session_service=session_svc)
```

### Example 2 — sliding-window compaction

```python
from google.adk.apps.app import App
from google.adk.apps._configs import EventsCompactionConfig

app = App(
    name="long-running-app",
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=10,  # compact after every 10 user turns
        overlap_size=2,          # include 2 turns of overlap between summaries
    ),
)
```

### Example 3 — token-threshold + sliding-window combined

```python
from google.adk.apps.app import App
from google.adk.apps._configs import EventsCompactionConfig

app = App(
    name="token-aware-app",
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(
        # Sliding-window trigger:
        compaction_interval=20,
        overlap_size=3,
        # Token-threshold trigger (fires post-invocation if prompt > 80K tokens):
        token_threshold=80_000,
        event_retention_size=10,  # keep last 10 raw events after token compaction
    ),
)
```

### Example 4 — custom summarizer

```python
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions, EventCompaction
from google.genai import types
from typing import Optional

class BulletSummarizer(BaseEventsSummarizer):
    """Summarizes events as simple bullet points without an LLM call."""

    async def maybe_summarize_events(
        self, *, events: list[Event]
    ) -> Optional[Event]:
        if not events:
            return None
        lines = []
        for ev in events:
            if ev.author == "user" and ev.content and ev.content.parts:
                text = " ".join(p.text or "" for p in ev.content.parts)
                if text.strip():
                    lines.append(f"• User asked: {text.strip()[:80]}")
            elif ev.is_final_response() and ev.content and ev.content.parts:
                text = " ".join(p.text or "" for p in ev.content.parts)
                if text.strip():
                    lines.append(f"• Agent replied: {text.strip()[:80]}")
        if not lines:
            return None
        summary_text = "\n".join(lines)
        compacted_content = types.Content(
            role="model",
            parts=[types.Part(text=summary_text)],
        )
        compaction = EventCompaction(
            start_timestamp=events[0].timestamp,
            end_timestamp=events[-1].timestamp,
            compacted_content=compacted_content,
        )
        return Event(
            author="user",
            actions=EventActions(compaction=compaction),
            invocation_id=Event.new_id(),
        )

from google.adk.apps._configs import EventsCompactionConfig

app = App(
    name="custom-compaction-app",
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5,
        overlap_size=1,
        summarizer=BulletSummarizer(),
    ),
)
```

### Example 5 — resumable app with `DatabaseSessionService`

```python
from google.adk.apps.app import App
from google.adk.apps._configs import ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService

app = App(
    name="resumable-app",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

# DatabaseSessionService uses SQLAlchemy; supports SQLite, PostgreSQL, MySQL, Spanner.
session_svc = DatabaseSessionService(db_url="sqlite+aiosqlite:///sessions.db")
runner = Runner(app=app, session_service=session_svc)
```

### Example 6 — context cache config (cost-reduction for repeated prompts)

```python
from google.adk.apps.app import App
from google.adk.agents.context_cache_config import ContextCacheConfig

app = App(
    name="cached-app",
    root_agent=root_agent,
    context_cache_config=ContextCacheConfig(
        cache_intervals=10,            # refresh cache every 10 invocations
        ttl_seconds=3600,              # cache lives for 1 hour
        min_tokens=4096,               # only cache if > 4K prompt tokens
    ),
)
```

---

## 10 · `Context` — unified CallbackContext / ToolContext API

**Source:** `google.adk.agents.context`

In 2.x, `CallbackContext` and `ToolContext` are both aliases for `Context`.
`Context` extends `ReadonlyContext` and exposes write methods for state, artifacts,
memory, auth credentials, and workflow scheduling.

### Inheritance chain

```
ReadonlyContext           # read-only properties: state (proxy), user_content,
                          #   invocation_id, agent_name, user_id, run_config
    └── Context           # + write methods: state delta, artifacts, memory,
                          #   auth, tool confirmation, workflow scheduling
```

### Read-only properties (from `ReadonlyContext`)

```python
ctx.invocation_id     # str — unique to this run_async() call
ctx.agent_name        # str — currently running agent's name
ctx.user_id           # str — from session
ctx.session           # Session — current session
# NOTE: ctx.state below is READ-ONLY when accessed from ReadonlyContext
# (returned as MappingProxyType). In the full Context subclass (callbacks
# and tools), state is overridden to return a mutable State object that
# tracks deltas — see "State mutations" below.
ctx.state             # MappingProxyType (ReadonlyContext) — immutable snapshot
ctx.user_content      # types.Content | None — message that started this invocation
ctx.run_config        # RunConfig | None — per-invocation config
ctx.get_credential(key)  # AuthCredential | None
```

### State mutations (write — `Context` subclass only)

`Context` overrides the `state` property to return a `State` object (from
`google.adk.sessions.state`). `State.__setitem__` writes to an in-memory
`state_delta` dict that is flushed to the session when the event is appended.
Direct `__setitem__` on a bare `ReadonlyContext.state` (a `MappingProxyType`)
raises `TypeError` — mutations are only valid inside callbacks and tools that
receive a full `Context`.

```python
# In a callback — ctx is a Context instance (not ReadonlyContext):
ctx.state["shopping_cart"] = ["item1", "item2"]   # writes to state delta
ctx.state["session_count"] = ctx.state.get("session_count", 0) + 1

# In a tool — tool_context is also a Context:
from google.adk.tools.tool_context import ToolContext

def update_user_pref(preference: str, tool_context: ToolContext) -> str:
    """Store a user preference."""
    tool_context.state["user:preferred_language"] = preference
    # Prefix "user:" → persists across sessions for this user
    return f"Saved preference: {preference}"
```

### State scope prefixes

| Prefix | Scope | Example key |
|---|---|---|
| `app:` | All users, all sessions | `app:feature_flags` |
| `user:` | This user, all their sessions | `user:preferred_theme` |
| `temp:` | This invocation only (not persisted) | `temp:scratch_pad` |
| *(none)* | This session only | `cart_items` |

### Artifacts: saving and loading

```python
import asyncio
from google.adk.tools.tool_context import ToolContext
from google.genai import types

async def save_report(content: str, tool_context: ToolContext) -> str:
    """Save a generated report as an artifact."""
    version = await tool_context.save_artifact(
        filename="report.txt",
        artifact=types.Part(
            inline_data=types.Blob(
                mime_type="text/plain",
                data=content.encode(),
            )
        ),
    )
    return f"Report saved as version {version}."

async def load_latest_report(tool_context: ToolContext) -> str:
    """Load the most recent report artifact."""
    artifact = await tool_context.load_artifact("report.txt")
    if artifact is None:
        return "No report found."
    return artifact.inline_data.data.decode()

async def load_pinned_report(version: int, tool_context: ToolContext) -> str:
    """Load a specific version of the report."""
    artifact = await tool_context.load_artifact("report.txt", version=version)
    if artifact is None:
        return f"Version {version} not found."
    return artifact.inline_data.data.decode()
```

### Memory: adding and searching

```python
from google.adk.tools.tool_context import ToolContext
from google.adk.memory.base_memory_service import MemoryEntry
from google.genai import types

async def remember_fact(fact: str, tool_context: ToolContext) -> str:
    """Add a fact to long-term memory."""
    await tool_context.add_memory(
        memories=[MemoryEntry(
            content=types.Content(
                role="user",
                parts=[types.Part(text=fact)],
            ),
            author="tool",
        )]
    )
    return f"Remembered: {fact}"

async def search_memory(query: str, tool_context: ToolContext) -> str:
    """Search long-term memory for relevant facts."""
    result = await tool_context.search_memory(query=query)
    if not result.memories:
        return "Nothing found."
    texts = []
    for m in result.memories:
        if m.content and m.content.parts:
            texts.append(" ".join(p.text or "" for p in m.content.parts))
    return "\n".join(texts) if texts else "Nothing found."
```

### Auth: requesting and reading credentials

```python
from google.adk.tools.tool_context import ToolContext
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_schemes import OAuthGrantType

async def call_github_api(repo: str, tool_context: ToolContext) -> str:
    """Fetch GitHub repo info with OAuth2 credential."""
    cred = tool_context.get_auth_response(auth_config)
    if cred is None:
        # Trigger the auth flow — the runner will pause and ask the user
        tool_context.request_credential(auth_config)
        return "Please complete authentication."
    token = cred.oauth2.access_token
    # … make API call with token …
    return f"Fetched repo {repo}"
```

### Tool confirmation (HITL)

```python
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.tool_confirmation import ToolConfirmation

async def delete_file(path: str, tool_context: ToolContext) -> str:
    """Delete a file, requiring explicit user confirmation."""
    confirmation: ToolConfirmation | None = tool_context.tool_confirmation
    if confirmation is None or not confirmation.confirmed:
        # Pause for user approval; resume once confirmed
        tool_context.request_confirmation(
            hint=f"Delete file at '{path}'?",
            payload={"path": path},
        )
        return "Waiting for confirmation."
    # Confirmed — perform the deletion
    import os
    os.remove(path)
    return f"Deleted {path}"
```

### Workflow routing from a tool

```python
from google.adk.tools.tool_context import ToolContext

def decide_path(user_intent: str, tool_context: ToolContext) -> str:
    """Route workflow to a different branch based on intent."""
    if "urgent" in user_intent.lower():
        tool_context.actions.route = "escalation_node"
    else:
        tool_context.actions.route = "standard_node"
    return f"Routing to {'escalation' if 'urgent' in user_intent.lower() else 'standard'} path."
```

### Complete callback example — before_agent with context inspection

```python
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from typing import Optional
from google.genai import types

async def before_agent_cb(callback_context: Context) -> Optional[types.Content]:
    """Enrich agent context before each agent call."""
    # Read current state
    turns = callback_context.state.get("turn_count", 0)
    callback_context.state["turn_count"] = turns + 1

    # Load user preferences from persistent state
    lang = callback_context.state.get("user:preferred_language", "en")

    # Optionally skip this agent by returning content directly
    if turns >= 100:
        return types.Content(
            role="model",
            parts=[types.Part(text="Session limit reached. Please start a new chat.")]
        )
    return None  # continue with normal agent execution

agent = LlmAgent(
    name="context_aware",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant. Respond in the user's language.",
    before_agent_callback=before_agent_cb,
)
```

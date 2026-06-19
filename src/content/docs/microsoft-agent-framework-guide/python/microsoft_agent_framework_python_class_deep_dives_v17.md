---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 17"
description: "Source-verified deep dives into 10 class groups new in agent-framework 1.9.0: ToolApprovalMiddleware+ToolApprovalRule+ToolApprovalState (interactive tool approval with standing rules), AgentLoopMiddleware+JudgeVerdict (LLM-judged self-improvement loops), SamplingApprovalCallback+MCP sampling security (confused-deputy mitigation), to_prompt_agent (Foundry PromptAgent publisher), FoundryEmbeddingClient+FoundryEmbeddingOptions (multimodal embeddings), ContentUnderstandingContextProvider+AnalysisSection+DocumentStatus (Azure Content Understanding integration), FileSearchConfig+FileSearchBackend (vector store upload/delete backends), AgentFrameworkTracer (Agent Lightning OTel bridge), TaskRunner+patch_env_set_state (Tau2 airline benchmark harness), new FoundryChatClient hosted tool factories."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 40
---

import { Aside } from '@astrojs/starlight/components';

# Microsoft Agent Framework Python — Class Deep Dives Vol. 17

Verified against **agent-framework 1.9.0** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source. Sub-packages introspected: `agent_framework._harness`, `agent_framework._mcp`,
`agent_framework.foundry`, `agent_framework.lab.lightning`, `agent_framework.lab.tau2`,
`agent_framework_azure_contentunderstanding`.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, middleware ABCs, compaction, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — harness providers, compaction strategies, `WorkflowViz`, MCP transports
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — message/chat types, `ResponseStream`, `AgentContext`, functional workflows, `SkillsSource`, eval model, tokenizer, `ConversationSplit`
- [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) — `Executor`, `AgentExecutor`, edge groups, `Runner`, `SessionContext`, `AgentSession`, `BaseChatClient`, `SecretString`, `WorkflowCheckpoint`, exceptions
- [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) — feature staging, `WorkflowRunState`, `WorkflowExecutor`, `AgentResponse`, embedding clients, `FunctionInvocationConfiguration`, `ClassSkill`, `Annotation`, capability protocols, middleware layers
- [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) — `ContextProvider`, `BackgroundTaskInfo`, orchestration builders, `AgentFactory`, `SecureAgentConfig`, `ObservabilitySettings`
- [Vol. 8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) — file store hierarchy, `FileAccessProvider`, `MCPSkill`, `ToolMode`, eval helpers, `ChatContext`, `WorkflowAgent`, compaction, history providers, skills composition
- [Vol. 9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) — `OllamaChatClient`, `PurviewPolicyMiddleware`, `DurableAIAgent`, `GitHubCopilotAgent`, `HyperlightExecuteCodeTool`, `Mem0ContextProvider`, Redis providers, Magentic internals, `FileSkillsSource`
- [Vol. 10](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10/) — `Workflow`, `InProcRunnerContext`, `FunctionExecutor`, `FunctionInvocationLayer`, memory harness, todo harness, `DeduplicatingSkillsSource`, `SkillsProvider`, `MCPTaskOptions`, `InMemoryCheckpointStorage`, `BaseAgent`
- [Vol. 11](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v11/) — telemetry layers, `Edge`+`EdgeGroup` primitives, `Case`+`Default`, `EdgeRunner` hierarchy, `ExecutionContext`, `WorkflowGraphValidator`, `MCPTool`, serialization mixin, `Evaluator`, `PerServiceCallHistoryPersistingMiddleware`
- [Vol. 12](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v12/) — Skills ABCs, `FileSkill`, `InlineSkillResource`+`InlineSkillScript`, `FileSkillScript`+`SkillScriptRunner`, `SupportsAgentRun`, `RunnerContext`, edge-routing descriptors, `WorkflowValidationError` hierarchy, `A2AAgent`+`A2AExecutor`, exception leaf classes
- [Vol. 13](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v13/) — OpenAI Responses/Completions/Embedding clients, Anthropic + Claude agent clients, multi-cloud Claude variants, group-chat + handoff + Magentic orchestration internals, declarative HTTP/MCP/approval handlers
- [Vol. 14](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v14/) — `State` (superstep cache), `OutputDesignation`, `MessageType`+`WorkflowMessage` internals, `DictConvertible` mixin, middleware pipeline hierarchy, `MiddlewareDict`, `FunctionRequestResult`, `OtelAttr`, security policy classes
- [Vol. 15](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v15/) — AG-UI client layer, AG-UI protocol wrappers, ChatKit, DevServer, GAIA benchmark, CopilotStudioAgent, AzureAISearchContextProvider, CosmosHistoryProvider, Durable external layer, AgentFunctionApp
- [Vol. 16](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v16/) — `FoundryAgent`+`FoundryAgentOptions`, `FoundryLocalClient`, `FoundryMemoryProvider`, `FoundryEvals`+`GeneratedEvaluatorRef`, `BedrockChatClient`, `BedrockEmbeddingClient`, `MagenticManagerBase`, `BaseGroupChatOrchestrator`+events, `AgentRequestInfoResponse`+`CacheProvider`, Purview exception hierarchy+`acquire_token`

This volume covers **ten new class groups** introduced in **1.9.0**, including the
interactive tool-approval harness, a self-improving loop middleware with LLM judging,
MCP sampling security controls, new Foundry hosting surface tools, multimodal embedding
support, Azure Content Understanding integration, and the Agent Lightning RL bridge.

---

## Table of contents

1. [`ToolApprovalMiddleware` + `ToolApprovalRule` + `ToolApprovalState` + `create_always_approve_tool_response` + `create_always_approve_tool_with_arguments_response`](#1-toolapprovalmiddleware)
2. [`AgentLoopMiddleware` + `JudgeVerdict` + `todos_remaining` + `background_tasks_running`](#2-agentloopmiddleware)
3. [`SamplingApprovalCallback` + MCP sampling security parameters](#3-samplingapprovalcallback)
4. [`to_prompt_agent`](#4-to_prompt_agent)
5. [`FoundryEmbeddingClient` + `FoundryEmbeddingOptions` + `FoundryEmbeddingSettings` + `RawFoundryEmbeddingClient`](#5-foundryembeddingclient)
6. [`ContentUnderstandingContextProvider` + `AnalysisSection` + `DocumentStatus`](#6-contentunderstandingcontextprovider)
7. [`FileSearchConfig` + `FileSearchBackend` + `OpenAIFileSearchBackend` + `FoundryFileSearchBackend`](#7-filesearchconfig)
8. [`AgentFrameworkTracer`](#8-agentframeworktracer)
9. [`TaskRunner` (lab.tau2) + `patch_env_set_state` + `unpatch_env_set_state`](#9-taskrunner-labtau2)
10. [New `FoundryChatClient` hosted tool factories](#10-new-foundrychatclient-hosted-tool-factories)

---

## 1. `ToolApprovalMiddleware`

**Module:** `agent_framework._harness._tool_approval` (exported from `agent_framework`)  
**Feature stage:** `@experimental`

<Aside type="tip">
This is the **first volume covering agent-framework 1.9.0**. All ten class groups
documented here are new in this minor release. The upgrade is backward-compatible;
no existing APIs were removed.
</Aside>

`ToolApprovalMiddleware` is an `AgentMiddleware` that intercepts function-call content
before execution and routes it through a human-in-the-loop approval queue stored in
the agent's `AgentSession`. Auto-approval rules can short-circuit the queue for trusted
tool/argument combinations, surfacing only genuinely novel calls to the user. Both
`ToolApprovalRule` and `ToolApprovalState` implement `SerializationMixin`, so they
round-trip cleanly through checkpoint storage.

### Constructor reference

```
ToolApprovalMiddleware(
    source_id: str = "tool_approval",          # DEFAULT_TOOL_APPROVAL_SOURCE_ID
    auto_approval_rules: Sequence[ToolApprovalRuleCallback] | None = None,
)

# Type alias for rule callbacks
ToolApprovalRuleCallback = Callable[[Content], bool | Awaitable[bool]]
```

`ToolApprovalMiddleware` **requires** an `AgentSession`. Calling `agent.run(...)` without
a session raises `RuntimeError("ToolApprovalMiddleware requires an AgentSession.")`.

### `ToolApprovalRule`

```
ToolApprovalRule(
    tool_name: str,               # raises ValueError if empty after strip()
    arguments: dict | None = None, # None = approve any call; {} = approve no-argument calls only
    *,
    server_label: str | None = None,  # restrict to a specific hosted tool server
)
```

### `ToolApprovalState`

Stored in the session state under `source_id`. Implements `SerializationMixin`.

```python
class ToolApprovalState:
    rules: list[ToolApprovalRule]
    queued_approval_requests: list[Content]
    collected_approval_responses: list[Content]
```

### Helper functions

```
create_always_approve_tool_response(request, *, reason=None)
    # Sets additional_properties["tool_approval"]["always_approve"] = "tool"
    # Creates a standing rule that approves ALL future calls to this tool

create_always_approve_tool_with_arguments_response(request, *, reason=None)
    # Sets scope "tool_with_arguments"
    # Creates a standing rule for this exact argument combination only
    # Arguments are canonicalised via json.dumps(..., sort_keys=True, separators=(",", ":"), default=str)
```

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| Session required | Raises `RuntimeError` if `context.session is None` |
| Default `source_id` | `DEFAULT_TOOL_APPROVAL_SOURCE_ID = "tool_approval"` |
| Queue drain | When multiple approvals arrive simultaneously, only the first is surfaced; extras go into `ToolApprovalState.queued_approval_requests`; re-invoking the agent drains one per call |
| `arguments=None` | Auto-approves any call to the tool regardless of arguments |
| `arguments={}` | Auto-approves only calls with no arguments |
| `server_label` | Restricts the rule to a specific hosted tool server; read from `function_call.additional_properties.get("server_label")` |
| Canonical argument matching | `json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)` ensures dict key order does not affect matching |
| Serialisation | Both `ToolApprovalRule` and `ToolApprovalState` expose `to_dict()` / `from_dict()` for checkpoint persistence |
| `"tool"` vs `"tool_with_arguments"` scope | `create_always_approve_tool_response` grants a standing rule for the whole tool; `create_always_approve_tool_with_arguments_response` is scoped to exact arguments only |

**Example 1 — basic approval gate with a session:**

```python
import asyncio
from agent_framework import Agent, AgentSession, tool
from agent_framework import ToolApprovalMiddleware
from agent_framework.foundry import FoundryChatClient

@tool
def delete_record(record_id: str) -> str:
    """Delete a record by ID."""
    return f"Deleted record {record_id}"

async def main():
    middleware = ToolApprovalMiddleware()  # source_id="tool_approval"
    agent = Agent(
        client=FoundryChatClient(model="gpt-4o"),
        tools=[delete_record],
        middleware=[middleware],
        instructions="You are a data management assistant.",
    )
    session = AgentSession()

    # First run — the framework raises an approval request before executing the tool
    result = await agent.run("Delete record ABC-123.", session=session)

    # Inspect what is waiting for approval
    from agent_framework import ToolApprovalState
    state: ToolApprovalState = session.state.get("tool_approval")
    if state and state.queued_approval_requests:
        print(f"{len(state.queued_approval_requests)} request(s) queued for approval.")
    print(result.text)

asyncio.run(main())
```

**Example 2 — auto-approval rule for a trusted read-only tool:**

```python
import asyncio
from agent_framework import Agent, AgentSession, tool
from agent_framework import ToolApprovalMiddleware, ToolApprovalRule
from agent_framework.foundry import FoundryChatClient

@tool
def get_record(record_id: str) -> dict:
    """Retrieve a record by ID."""
    return {"id": record_id, "status": "active"}

@tool
def delete_record(record_id: str) -> str:
    """Delete a record by ID."""
    return f"Deleted record {record_id}"

def auto_approve_reads(content) -> bool:
    """Auto-approve all get_record calls; require manual approval for delete_record."""
    rule = ToolApprovalRule(tool_name="get_record")  # None arguments = approve any call
    # Match by checking tool name from content's function call properties
    return getattr(getattr(content, "function_call", None), "name", None) == "get_record"

async def main():
    middleware = ToolApprovalMiddleware(
        auto_approval_rules=[auto_approve_reads],
    )
    agent = Agent(
        client=FoundryChatClient(model="gpt-4o"),
        tools=[get_record, delete_record],
        middleware=[middleware],
    )
    session = AgentSession()
    # get_record is auto-approved; delete_record surfaces for human review
    result = await agent.run("Retrieve record XYZ-999.", session=session)
    print(result.text)

asyncio.run(main())
```

**Example 3 — create a standing "always approve" rule and checkpoint round-trip:**

```python
import asyncio
from agent_framework import Agent, AgentSession, tool
from agent_framework import ToolApprovalMiddleware, ToolApprovalState
from agent_framework import create_always_approve_tool_response

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

async def main():
    middleware = ToolApprovalMiddleware()
    agent = Agent(
        client=FoundryChatClient(model="gpt-4o"),
        tools=[send_email],
        middleware=[middleware],
        instructions="You are an email assistant.",
    )
    session = AgentSession()

    # Simulate: user approves the first request and clicks "Always approve this tool"
    # In a real UI you would pass the actual request Content object here.
    # create_always_approve_tool_response sets scope "tool" — all future send_email
    # calls are auto-approved for this session without prompting again.
    from agent_framework.foundry import FoundryChatClient  # noqa: re-import for clarity

    # Demonstrate serialisation round-trip of ToolApprovalState
    state = ToolApprovalState(rules=[], queued_approval_requests=[], collected_approval_responses=[])
    state_dict = state.to_dict()
    restored = ToolApprovalState.from_dict(state_dict)
    print(f"Round-trip OK: rules={restored.rules}, queued={restored.queued_approval_requests}")

asyncio.run(main())
```

---

## 2. `AgentLoopMiddleware`

**Module:** `agent_framework._harness._loop` (exported from `agent_framework`)  
**Feature stage:** `@experimental`

`AgentLoopMiddleware` wraps an agent in a self-improvement loop: after each iteration it
calls a configurable `should_continue` function (or an LLM judge via `with_judge`) to
decide whether the agent should keep working. A feedback string from the judge is injected
into the next iteration via `next_message` and recorded via `record_feedback`. The loop
respects `max_iterations` as a hard ceiling, short-circuiting before calling the
potentially expensive judge once the cap fires.

### Constructor reference

```
AgentLoopMiddleware(
    should_continue: ShouldContinueCallable,
    max_iterations: int | None = 10,          # DEFAULT_MAX_ITERATIONS; raises ValueError if < 1
    next_message: ... | None = None,          # receives (iteration, last_result, ..., feedback=)
    record_feedback: ... | None = None,       # defaults to last_result.text.strip() when None
    inject_progress: bool = True,             # inject progress log into next iteration context
    fresh_context: bool = False,              # restore session snapshot before each iteration
    return_final_only: bool = False,          # False = all iterations aggregated; True = last only
    additional_instructions: str | None = None,
)
```

`should_continue` may return a plain `bool` or a `(bool, str | None)` tuple. When a
string is returned as the second element it becomes the `feedback` keyword argument
passed to `next_message` and `record_feedback`.

### `DEFAULT_NEXT_MESSAGE`

```
"Continue working on the task. If it is complete, say so."
```

### `with_judge` class method

```
AgentLoopMiddleware.with_judge(
    judge_client,
    *,
    criteria: str | None = None,
    instructions: str | None = None,
    max_iterations: int = 5,           # DEFAULT_JUDGE_MAX_ITERATIONS
    ...
) -> AgentLoopMiddleware
```

The judge is called with `JudgeVerdict` as the structured output type. Fallback text
markers `"VERDICT: DONE"` / `"VERDICT: MORE"` are used when structured output is not
honoured. `"MORE"` wins on ambiguity to keep the loop going. `criteria` is injected both
as `additional_instructions` to the agent and rendered into the judge via the
`{{criteria}}` placeholder (`CRITERIA_PLACEHOLDER`).

### `todos_remaining` / `background_tasks_running`

```
todos_remaining(provider) -> ShouldContinueCallable
    # Reads provider.store.load_items(session, source_id=provider.source_id)
    # Continues while any item is not complete; returns False if session is None

background_tasks_running(provider) -> ShouldContinueCallable
    # Reads session.state.get(provider.source_id)
    # Continues while any task has BackgroundTaskStatus.RUNNING
```

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| `max_iterations < 1` | Raises `ValueError("max_iterations must be None or a positive integer (>= 1).")` |
| Cap short-circuits | `max_iterations` fires **before** calling `should_continue` — the judge is not invoked on the final cap iteration |
| `inject_progress` with session | Only the **latest** progress entry is injected (earlier ones already in session history); without a session the **full log** is injected |
| `fresh_context=True` | Session snapshot taken via `session.to_dict()` once before the loop; restored via `AgentSession.from_dict(snapshot)` before each subsequent iteration, copying `service_session_id` and `state` back in-place |
| `return_final_only=False` | Non-streaming run returns an aggregated response of all iterations plus nudge messages |
| Streaming | Each iteration is yielded as it completes; nudge messages are injected as `user` updates between iterations |
| `record_feedback=None` | Defaults to `last_result.text.strip()` |
| Loop kwargs | `iteration` (1-based after first run), `last_result`, `messages`, `original_messages`, `session`, `agent`, `progress` (copy), `feedback` |

**Example 1 — simple loop with a fixed iteration cap:**

```python
import asyncio
from agent_framework import Agent, AgentLoopMiddleware
from agent_framework.foundry import FoundryChatClient

def should_keep_going(*, last_result, iteration, **_):
    """Continue until the agent explicitly says 'DONE' or we hit the cap."""
    text = (last_result.text or "").upper()
    return "DONE" not in text

async def main():
    loop_mw = AgentLoopMiddleware(
        should_continue=should_keep_going,
        max_iterations=5,
    )
    agent = Agent(
        client=FoundryChatClient(model="gpt-4o"),
        middleware=[loop_mw],
        instructions=(
            "Work through the task step by step. "
            "When the task is complete, end your message with 'DONE'."
        ),
    )
    result = await agent.run("List and briefly explain 3 sorting algorithms.")
    print(result.text)

asyncio.run(main())
```

**Example 2 — LLM judge loop with criteria:**

```python
import asyncio
from agent_framework import Agent, AgentLoopMiddleware
from agent_framework.foundry import FoundryChatClient

async def main():
    chat_client = FoundryChatClient(model="gpt-4o")
    judge_client = FoundryChatClient(model="gpt-4o-mini")

    loop_mw = AgentLoopMiddleware.with_judge(
        judge_client,
        criteria=(
            "The response must include: (1) a clear problem statement, "
            "(2) at least two proposed solutions, (3) a recommendation with rationale."
        ),
        max_iterations=4,
    )
    agent = Agent(
        client=chat_client,
        middleware=[loop_mw],
        instructions="You are a technical architect.",
    )
    result = await agent.run(
        "How should we handle database connection pooling in a high-traffic web service?"
    )
    print(result.text)

asyncio.run(main())
```

**Example 3 — fresh-context loop with streaming and todo-based continuation:**

```python
import asyncio
from agent_framework import Agent, AgentSession, AgentLoopMiddleware
from agent_framework import TodoContextProvider, background_tasks_running
from agent_framework.foundry import FoundryChatClient

async def main():
    client = FoundryChatClient(model="gpt-4o")
    todo_provider = TodoContextProvider()

    loop_mw = AgentLoopMiddleware(
        should_continue=lambda *, last_result, **_: (
            "task complete" not in (last_result.text or "").lower()
        ),
        max_iterations=6,
        fresh_context=True,        # restore session snapshot before each iteration
        return_final_only=True,    # only the last iteration matters to the caller
        inject_progress=True,
    )
    agent = Agent(
        client=client,
        middleware=[loop_mw],
        context_providers=[todo_provider],
        instructions="Complete all outstanding tasks. Say 'task complete' when done.",
    )
    session = AgentSession()

    # Streaming: each iteration is yielded as it arrives
    stream = agent.run("Research and summarise the top 3 Python web frameworks.", stream=True, session=session)
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()

asyncio.run(main())
```

---

## 3. `SamplingApprovalCallback`

**Module:** `agent_framework._mcp` (exported from `agent_framework`)

The `SamplingApprovalCallback` type alias and accompanying per-tool security parameters
give fine-grained control over MCP server-initiated `sampling/createMessage` requests.
Prior to 1.9.0, all sampling requests were silently auto-approved, creating a
confused-deputy risk where a compromised MCP server could covertly prompt the LLM.
In 1.9.0 this is **denied by default**.

### Type alias

```python
SamplingApprovalCallback = Callable[
    ["types.CreateMessageRequestParams"],
    "bool | Coroutine[Any, Any, bool]"
]
```

### New constructor parameters on all three MCP tool classes

The following parameters were added to `MCPStdioTool`, `MCPStreamableHTTPTool`, and
`MCPWebsocketTool`:

```
sampling_approval_callback: SamplingApprovalCallback | None = None
    # None (default) = deny all sampling requests
    # Pass lambda params: True to restore legacy auto-approve behaviour

sampling_max_tokens: int | None = _DEFAULT_SAMPLING_MAX_TOKENS
    # Cap on server-requested maxTokens; effective = min(requested, cap)
    # None disables the cap entirely

sampling_max_requests: int | None = _DEFAULT_SAMPLING_MAX_REQUESTS
    # Maximum approved sampling requests per session connection
    # Counter resets on reconnect; None disables the limit
```

The instance counter `_sampling_request_count` tracks how many sampling requests have
been approved in the current connection lifetime.

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| Secure by default | When `sampling_approval_callback=None`, every `sampling/createMessage` is **denied** |
| Legacy opt-in | Pass `lambda params: True` as an explicit choice to restore auto-approve |
| Token cap | Effective `maxTokens = min(server_requested, sampling_max_tokens)` when cap is set |
| Request limit | Once `_sampling_request_count >= sampling_max_requests`, further requests are denied for the rest of the connection |
| Counter reset | `_sampling_request_count` resets to 0 on each new server connection |
| Async callback | The callback may be a coroutine; `MCPStdioTool` awaits it before forwarding the sampling request |

**Example 1 — deny all sampling (secure default, explicit):**

```python
import asyncio
from agent_framework import Agent
from agent_framework import MCPStdioTool
from agent_framework.foundry import FoundryChatClient

async def main():
    # sampling_approval_callback=None is the default — shown here for clarity
    mcp_tool = MCPStdioTool(
        command="uvx",
        args=["my-mcp-server"],
        sampling_approval_callback=None,    # deny all server-initiated sampling
        sampling_max_tokens=4096,
        sampling_max_requests=10,
    )
    agent = Agent(
        client=FoundryChatClient(model="gpt-4o"),
        tools=[mcp_tool],
        instructions="Use the MCP server to answer questions.",
    )
    result = await agent.run("What data does the server have?")
    print(result.text)

asyncio.run(main())
```

**Example 2 — conditional sampling approval based on request parameters:**

```python
import asyncio
from agent_framework import Agent
from agent_framework import MCPStreamableHTTPTool, SamplingApprovalCallback
from agent_framework.foundry import FoundryChatClient

ALLOWED_SYSTEM_PROMPTS = {"You are a helpful assistant.", "Summarise the following."}

def review_sampling_request(params) -> bool:
    """Only approve sampling if the server provides a known safe system prompt."""
    messages = getattr(params, "messages", []) or []
    system_msgs = [m for m in messages if getattr(m, "role", None) == "system"]
    if not system_msgs:
        return False  # no system prompt — deny
    system_text = getattr(system_msgs[0].content, "text", "") if system_msgs else ""
    return system_text in ALLOWED_SYSTEM_PROMPTS

async def main():
    mcp_tool = MCPStreamableHTTPTool(
        url="https://my-mcp-server.example.com/mcp",
        sampling_approval_callback=review_sampling_request,
        sampling_max_tokens=2048,
        sampling_max_requests=5,
    )
    agent = Agent(
        client=FoundryChatClient(model="gpt-4o"),
        tools=[mcp_tool],
    )
    result = await agent.run("Summarise the latest reports from the server.")
    print(result.text)

asyncio.run(main())
```

**Example 3 — async approval callback with audit logging:**

```python
import asyncio
import logging
from agent_framework import Agent
from agent_framework import MCPWebsocketTool
from agent_framework.foundry import FoundryChatClient

logger = logging.getLogger("sampling_audit")

async def audit_and_approve(params) -> bool:
    """Log every sampling request and approve those under 1000 max_tokens."""
    requested_max = getattr(params, "max_tokens", 0) or 0
    logger.info("MCP sampling request: max_tokens=%d", requested_max)
    approved = requested_max <= 1000
    logger.info("Sampling %s", "APPROVED" if approved else "DENIED")
    return approved

async def main():
    logging.basicConfig(level=logging.INFO)
    mcp_tool = MCPWebsocketTool(
        url="wss://my-mcp-server.example.com/ws",
        sampling_approval_callback=audit_and_approve,
        sampling_max_tokens=1000,   # hard cap regardless of callback outcome
        sampling_max_requests=20,
    )
    agent = Agent(
        client=FoundryChatClient(model="gpt-4o"),
        tools=[mcp_tool],
    )
    result = await agent.run("Query the server for today's metrics.")
    print(result.text)

asyncio.run(main())
```

---

## 4. `to_prompt_agent`

**Module:** `agent_framework.foundry` (from `agent_framework_foundry._to_prompt_agent`)  
**Feature stage:** `@experimental(feature_id=ExperimentalFeature.TO_PROMPT_AGENT)`  
**Install:** `pip install agent-framework[foundry]`

`to_prompt_agent` converts a locally-defined `Agent` into a `PromptAgentDefinition` that
can be published to Azure AI Foundry via `AIProjectClient.agents.create_version(...)`.
It translates chat options, tool declarations, and response format settings into the
Foundry representation, surfacing incompatibilities (local MCP tools, missing model) as
early `ValueError`/`TypeError` rather than at deployment time.

### Signature

```
to_prompt_agent(
    agent: Agent,
    *,
    structured_inputs=None,
    rai_config=None,
) -> PromptAgentDefinition
```

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| Client type check | Raises `TypeError` if `agent.client` is not a `RawFoundryChatClient` subclass |
| Model resolution | `agent.default_options.get("model") or agent.client.model`; raises `ValueError` if neither is set |
| Options translated | `temperature`, `top_p`, `reasoning`, `tool_choice`, `response_format`/`text`/`verbosity`; uses `_prepare_response_and_text_format` for format consistency |
| Options ignored | `include`, `prompt`, `store`, and other `OpenAIChatOptions` keys with no `PromptAgentDefinition` equivalent |
| `FunctionTool` | Converted to Foundry `FunctionTool` declaration (schema only; no Python execution wired server-side) |
| `ProjectsTool` | Hosted tool instances passed through unchanged |
| Local `MCPTool` | Raises `ValueError` — use `FoundryChatClient.get_mcp_tool()` for hosted MCP instead |
| `tool_choice` | Dropped when no tools are present on the definition (mirrors regular request path) |
| Return value | `PromptAgentDefinition`; pass to `AIProjectClient.agents.create_version(...)` to publish |

**Example 1 — basic function-tool agent published to Foundry:**

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient, to_prompt_agent
from azure.ai.projects.aio import AIProjectClient
from azure.identity import DefaultAzureCredential

@tool
def get_weather(city: str) -> str:
    """Return the weather for a given city."""
    return f"Sunny, 22C in {city}"

async def main():
    client = FoundryChatClient(model="gpt-4o")
    agent = Agent(
        client=client,
        tools=[get_weather],
        instructions="You are a weather assistant.",
        default_options={"temperature": 0.3},
    )

    definition = to_prompt_agent(agent)  # raises TypeError/ValueError early if invalid
    print(f"Definition model: {definition.model}")
    print(f"Tool count: {len(definition.tools or [])}")

    # Publish to Foundry
    project_client = AIProjectClient(
        endpoint="https://my-proj.services.ai.azure.com",
        credential=DefaultAzureCredential(),
    )
    async with project_client:
        version = await project_client.agents.create_version(
            agent_name="weather-assistant",
            body=definition,
        )
        print(f"Published version: {version.version}")

asyncio.run(main())
```

**Example 2 — structured output agent with response format translation:**

```python
import asyncio
from pydantic import BaseModel
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient, to_prompt_agent

class AnalysisResult(BaseModel):
    sentiment: str
    confidence: float
    key_topics: list[str]

async def main():
    client = FoundryChatClient(model="gpt-4o")
    agent = Agent(
        client=client,
        instructions="Analyse text and return structured results.",
        default_options={
            "response_format": AnalysisResult,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    )
    definition = to_prompt_agent(agent)
    # response_format is translated via _prepare_response_and_text_format
    print(f"Response format set: {definition.response_format is not None}")
    print(f"Temperature: {definition.temperature}")

asyncio.run(main())
```

**Example 3 — hosted tools pass through; local MCP raises ValueError:**

```python
import asyncio
from agent_framework import Agent
from agent_framework import MCPStdioTool
from agent_framework.foundry import FoundryChatClient, to_prompt_agent

async def main():
    client = FoundryChatClient(model="gpt-4o")

    # Hosted MCP tool — passes through unchanged
    hosted_mcp = client.get_mcp_tool(
        server_url="https://my-mcp.example.com/mcp",
        server_label="my-mcp",
    )
    agent_with_hosted = Agent(client=client, tools=[hosted_mcp])
    definition = to_prompt_agent(agent_with_hosted)
    print(f"Hosted MCP included: {len(definition.tools or [])} tool(s)")

    # Local MCP tool — raises ValueError at conversion time, not deployment time
    local_mcp = MCPStdioTool(command="uvx", args=["my-local-server"])
    agent_with_local = Agent(client=client, tools=[local_mcp])
    try:
        to_prompt_agent(agent_with_local)
    except ValueError as exc:
        print(f"Expected error: {exc}")

asyncio.run(main())
```

---

## 5. `FoundryEmbeddingClient`

**Module:** `agent_framework.foundry` (from `agent_framework_foundry._embedding_client`)  
**Install:** `pip install agent-framework[foundry]`

`FoundryEmbeddingClient` adds `EmbeddingTelemetryLayer` on top of
`RawFoundryEmbeddingClient`, which accepts both `str` (text) and `Content` (image) inputs
in a single batch. Mixed batches are transparently split and dispatched to the appropriate
underlying client (`EmbeddingsClient` for text, `ImageEmbeddingsClient` for images), with
results reassembled in original input order.

### Class hierarchy

```
BaseEmbeddingClient[Content | str, list[float], FoundryEmbeddingOptionsT]
  └── RawFoundryEmbeddingClient
        └── FoundryEmbeddingClient (adds EmbeddingTelemetryLayer)
```

### Constructor reference (`RawFoundryEmbeddingClient` / `FoundryEmbeddingClient`)

```
FoundryEmbeddingClient(
    model: str | None = None,           # or FOUNDRY_EMBEDDING_MODEL
    *,
    image_model: str | None = None,     # or FOUNDRY_IMAGE_EMBEDDING_MODEL; falls back to model
    endpoint: str | None = None,        # or FOUNDRY_MODELS_ENDPOINT
    api_key: str | None = None,         # or FOUNDRY_MODELS_API_KEY
    additional_properties: dict | None = None,
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
)
```

### `FoundryEmbeddingSettings` (env var mapping)

| Setting | Env var |
|---------|---------|
| `models_endpoint` | `FOUNDRY_MODELS_ENDPOINT` |
| `models_api_key` | `FOUNDRY_MODELS_API_KEY` |
| `embedding_model` | `FOUNDRY_EMBEDDING_MODEL` |
| `image_embedding_model` | `FOUNDRY_IMAGE_EMBEDDING_MODEL` |

### `FoundryEmbeddingOptions`

Extends `EmbeddingGenerationOptions` with Foundry-specific fields:

| Field | Type | Notes |
|-------|------|-------|
| `input_type` | `str` | Embedding input type hint (e.g., `"query"`, `"document"`) |
| `image_model` | `str` | Per-call image model override; falls back to client-level `image_model` then `model` |
| `encoding_format` | `str` | Output format (e.g., `"float"`, `"base64"`) |
| `extra_parameters` | `dict` | Forwarded as additional JSON body fields |

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| Mixed batch handling | Text and image inputs are separated; dispatched to `EmbeddingsClient` and `ImageEmbeddingsClient` respectively; results reassembled in original order |
| Image detection | `media_type` prefix `"image/"` triggers `ImageEmbeddingsClient` dispatch (`_IMAGE_MEDIA_PREFIXES`) |
| `image_model` precedence | Per-call option → client-level `image_model` → client-level `model` |
| `service_url` property | Returns the endpoint URL |
| Async context manager | `async with FoundryEmbeddingClient(...) as emb:` closes both underlying httpx clients on exit |
| Return type | `GeneratedEmbeddings` carries embeddings + usage statistics |

**Example 1 — text embeddings for semantic similarity:**

```python
import asyncio
from agent_framework.foundry import FoundryEmbeddingClient

async def main():
    async with FoundryEmbeddingClient(
        model="text-embedding-3-large",
        endpoint="https://my-proj.services.ai.azure.com",
    ) as client:
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "The Eiffel Tower is in Paris.",
        ]
        result = await client.get_embeddings(texts)
        print(f"Embedded {len(result)} strings, dim={len(result[0].vector)}")
        print(f"Usage: {result.usage}")

asyncio.run(main())
```

**Example 2 — mixed text and image batch in one call:**

```python
import asyncio
from agent_framework.foundry import FoundryEmbeddingClient, FoundryEmbeddingOptions
from agent_framework import Content, ImageContent

async def main():
    async with FoundryEmbeddingClient(
        model="text-embedding-3-large",
        image_model="image-embedding-v1",
        endpoint="https://my-proj.services.ai.azure.com",
    ) as client:
        # Mix text strings and Content objects in a single batch
        text_input = "A dog playing fetch in the park."
        image_input = ImageContent(
            media_type="image/png",
            data=b"\x89PNG...",   # PNG bytes
        )
        # Results are reassembled in original input order
        results = await client.get_embeddings(
            [text_input, image_input],
            options=FoundryEmbeddingOptions(input_type="document"),
        )
        print(f"Text embedding dim: {len(results[0].vector)}")
        print(f"Image embedding dim: {len(results[1].vector)}")

asyncio.run(main())
```

**Example 3 — per-call image model override and encoding format:**

```python
import asyncio
from agent_framework.foundry import FoundryEmbeddingClient, FoundryEmbeddingOptions

async def main():
    async with FoundryEmbeddingClient(
        model="text-embedding-3-small",      # default text model
        image_model="image-embedding-v1",    # default image model
        endpoint="https://my-proj.services.ai.azure.com",
    ) as client:
        # Override image_model per call and request base64 output
        options = FoundryEmbeddingOptions(
            image_model="image-embedding-v2",   # overrides client-level image_model
            encoding_format="base64",
            extra_parameters={"truncation": True},
        )
        texts = ["Retrieval augmented generation improves LLM accuracy."]
        result = await client.get_embeddings(texts, options=options)
        print(f"Encoding format applied, vector length: {len(result[0].vector)}")
        print(f"Endpoint: {client.service_url}")

asyncio.run(main())
```

---

## 6. `ContentUnderstandingContextProvider`

**Module:** `agent_framework_azure_contentunderstanding` (exported via `agent_framework.foundry`)  
**Install:** `pip install agent-framework-azure-contentunderstanding`

`ContentUnderstandingContextProvider` extends `ContextProvider` and integrates Azure AI
Content Understanding into the agent lifecycle. On `before_run` it detects and strips file
attachments from messages, submits them to the Content Understanding service (blocking or
deferred), and injects LLM-ready text into the agent's context. Two auto-registered tools
(`list_documents` and a file-content retrieval tool) let the agent query document status
and content at runtime.

### Constructor reference

```
ContentUnderstandingContextProvider(
    endpoint: str | None = None,          # or CONTENT_UNDERSTANDING_ENDPOINT
    api_key: str | None = None,           # or CONTENT_UNDERSTANDING_API_KEY
    analyzer_id: str | None = None,       # None = auto-detect from media type
    sections: list[AnalysisSection] = ["markdown"],  # "markdown" | "fields"
    deferred: bool = False,               # True = background analysis; False = blocking
    file_search_config: FileSearchConfig | None = None,
)
```

### `AnalysisSection`

```python
AnalysisSection = Literal["markdown", "fields"]
# "markdown" — full document text with tables rendered as HTML
# "fields"   — extracted typed fields with confidence scores
```

### `DocumentStatus`

```python
class DocumentStatus(str, Enum):
    ANALYZING = "analyzing"
    UPLOADING = "uploading"
    READY     = "ready"
    FAILED    = "failed"
```

### Auto-detected analyzer IDs

| Media type prefix | Analyzer |
|-------------------|----------|
| `"audio/"` | `"prebuilt-audioSearch"` |
| `"video/"` | `"prebuilt-videoSearch"` |
| (default) | `"prebuilt-documentSearch"` |

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| Registration | `session.add_context_provider(provider)` — hook-based, not passed at `Agent` construction |
| `before_run` | Detects + strips file attachments from messages; starts CU analysis; injects result via `extend_instructions` |
| `deferred=False` | Analysis completes before the agent call (blocking) |
| `deferred=True` | Analysis runs in the background; subsequent agent turns pick up completed results |
| `_render_for_llm()` | Calls `azure.ai.contentunderstanding.to_llm_input()` to produce YAML front matter + markdown |
| `list_documents` tool | Auto-registered; returns JSON of all tracked document states including `DocumentStatus` |
| `FileSearchConfig` | When provided, uploads analyzed markdown to a vector store and registers the `file_search` tool via `extend_tools` |
| `ContentUnderstandingSettings` env vars | `CONTENT_UNDERSTANDING_ENDPOINT`, `CONTENT_UNDERSTANDING_API_KEY` |

**Example 1 — basic document analysis injected into agent context:**

```python
import asyncio
from agent_framework import Agent, AgentSession
from agent_framework.foundry import (
    FoundryChatClient,
    ContentUnderstandingContextProvider,
    AnalysisSection,
)

async def main():
    provider = ContentUnderstandingContextProvider(
        endpoint="https://my-cu.cognitiveservices.azure.com",
        api_key="my-api-key",
        sections=["markdown"],   # full document text injected as markdown
        deferred=False,          # block until analysis completes
    )
    session = AgentSession()
    session.add_context_provider(provider)

    agent = Agent(
        client=FoundryChatClient(model="gpt-4o"),
        instructions="Answer questions based on the provided documents.",
    )
    # In a real scenario, messages would carry Content items with file attachments
    result = await agent.run("What are the key findings in the attached report?", session=session)
    print(result.text)

asyncio.run(main())
```

**Example 2 — audio/video content with deferred analysis:**

```python
import asyncio
from agent_framework import Agent, AgentSession
from agent_framework.foundry import (
    FoundryChatClient,
    ContentUnderstandingContextProvider,
    DocumentStatus,
)

async def main():
    provider = ContentUnderstandingContextProvider(
        endpoint="https://my-cu.cognitiveservices.azure.com",
        api_key="my-api-key",
        # analyzer_id=None: auto-detect "prebuilt-audioSearch" for audio/* media types
        deferred=True,  # background analysis; result injected on next agent turn
    )
    session = AgentSession()
    session.add_context_provider(provider)

    agent = Agent(
        client=FoundryChatClient(model="gpt-4o"),
        instructions="Summarise audio and video content when it becomes available.",
    )

    # Turn 1: attach audio file — analysis starts in background
    r1 = await agent.run("Analyse the attached meeting recording.", session=session)
    print("Turn 1:", r1.text)

    # Turn 2: CU analysis may now be complete; results are injected automatically
    r2 = await agent.run("What action items were mentioned?", session=session)
    print("Turn 2:", r2.text)

asyncio.run(main())
```

**Example 3 — extracted fields with confidence scores:**

```python
import asyncio
from agent_framework import Agent, AgentSession
from agent_framework.foundry import (
    FoundryChatClient,
    ContentUnderstandingContextProvider,
    AnalysisSection,
)

async def main():
    # "fields" section extracts typed fields with confidence scores
    # Useful for structured documents like invoices, receipts, or forms
    provider = ContentUnderstandingContextProvider(
        endpoint="https://my-cu.cognitiveservices.azure.com",
        api_key="my-api-key",
        analyzer_id="prebuilt-documentSearch",
        sections=["fields"],     # typed fields + confidence scores (not raw markdown)
        deferred=False,
    )
    session = AgentSession()
    session.add_context_provider(provider)

    agent = Agent(
        client=FoundryChatClient(model="gpt-4o"),
        instructions="Extract and summarise structured data from documents.",
    )
    result = await agent.run(
        "What is the invoice total and due date from the attached invoice?",
        session=session,
    )
    print(result.text)

asyncio.run(main())
```

---

## 7. `FileSearchConfig`

**Module:** `agent_framework_azure_contentunderstanding` (exported via `agent_framework.foundry`)  
**Install:** `pip install agent-framework-azure-contentunderstanding`

`FileSearchConfig` is a dataclass that pairs a `FileSearchBackend` with a vector store ID
and `file_search` tool definition. When passed to `ContentUnderstandingContextProvider`,
it causes analyzed document markdown to be uploaded to the vector store and the
`file_search` tool to be registered with the agent via `extend_tools`. Two concrete
backends are provided: `OpenAIFileSearchBackend` and `FoundryFileSearchBackend`, sharing
the same API surface via `_OpenAICompatBackend`.

### `FileSearchBackend` ABC

```python
class FileSearchBackend(ABC):
    async def upload_file(self, vector_store_id: str, filename: str, content: bytes) -> str:
        """Upload content to the vector store; return the file ID."""
        ...

    async def delete_file(self, file_id: str) -> None:
        """Delete a previously uploaded file."""
        ...
```

### `FileSearchConfig` dataclass

```python
@dataclass
class FileSearchConfig:
    backend: FileSearchBackend
    vector_store_id: str
    file_search_tool: Any

    @staticmethod
    def from_openai(client, *, vector_store_id: str, file_search_tool) -> "FileSearchConfig":
        """Wrap an OpenAI client in OpenAIFileSearchBackend."""
        ...

    @staticmethod
    def from_foundry(client, *, vector_store_id: str, file_search_tool) -> "FileSearchConfig":
        """Wrap a Foundry client in FoundryFileSearchBackend."""
        ...
```

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| `_OpenAICompatBackend` | Shared base for both backends; uses `client.files.create(file=(filename, io.BytesIO(content)), purpose=_FILE_PURPOSE)` then `create_and_poll` |
| `create_and_poll` | Waits for vector store indexing before returning — prevents empty search results on immediate query |
| `OpenAIFileSearchBackend._FILE_PURPOSE` | `"assistants"` |
| `FoundryFileSearchBackend._FILE_PURPOSE` | Foundry-specific purpose string |
| Vector store management | Caller is responsible for creating and managing the vector store; the backend only handles file upload and delete |
| `file_search_tool` | Caller creates the tool (e.g., `client.get_file_search_tool(vector_store_ids=[...])`) and passes it here |

**Example 1 — OpenAI-backed file search with Content Understanding:**

```python
import asyncio
from openai import AsyncOpenAI
from agent_framework import Agent, AgentSession
from agent_framework.foundry import (
    FoundryChatClient,
    ContentUnderstandingContextProvider,
    FileSearchConfig,
)

async def main():
    oai_client = AsyncOpenAI()

    # Create a vector store (caller-managed)
    vs = await oai_client.vector_stores.create(name="document-store")
    file_search_tool = {"type": "file_search", "vector_store_ids": [vs.id]}

    fs_config = FileSearchConfig.from_openai(
        oai_client,
        vector_store_id=vs.id,
        file_search_tool=file_search_tool,
    )

    provider = ContentUnderstandingContextProvider(
        endpoint="https://my-cu.cognitiveservices.azure.com",
        api_key="my-api-key",
        file_search_config=fs_config,  # uploads markdown to vector store + registers tool
    )
    session = AgentSession()
    session.add_context_provider(provider)

    agent = Agent(
        client=FoundryChatClient(model="gpt-4o"),
        instructions="Use file search to answer questions about uploaded documents.",
    )
    result = await agent.run("Find references to the Q3 budget in the attached documents.", session=session)
    print(result.text)

asyncio.run(main())
```

**Example 2 — Foundry-backed file search:**

```python
import asyncio
from agent_framework import Agent, AgentSession
from agent_framework.foundry import (
    FoundryChatClient,
    ContentUnderstandingContextProvider,
    FileSearchConfig,
)

async def main():
    foundry_client = FoundryChatClient(model="gpt-4o")

    # get_file_search_tool raises ValueError if vector_store_ids is missing
    file_search_tool = foundry_client.get_file_search_tool(vector_store_ids=["vs-abc123"])

    fs_config = FileSearchConfig.from_foundry(
        foundry_client,
        vector_store_id="vs-abc123",
        file_search_tool=file_search_tool,
    )
    provider = ContentUnderstandingContextProvider(
        endpoint="https://my-cu.cognitiveservices.azure.com",
        api_key="my-api-key",
        file_search_config=fs_config,
    )
    session = AgentSession()
    session.add_context_provider(provider)

    agent = Agent(
        client=foundry_client,
        instructions="Search documents and answer questions.",
    )
    result = await agent.run("What safety regulations are mentioned in the uploaded manual?", session=session)
    print(result.text)

asyncio.run(main())
```

**Example 3 — custom `FileSearchBackend` for a non-standard vector store:**

```python
import asyncio
import io
from agent_framework import Agent, AgentSession
from agent_framework.foundry import (
    FoundryChatClient,
    ContentUnderstandingContextProvider,
    FileSearchConfig,
    FileSearchBackend,
)

class MyVectorStoreBackend(FileSearchBackend):
    """Custom backend that calls an in-house vector store API."""

    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url
        self._api_key = api_key

    async def upload_file(self, vector_store_id: str, filename: str, content: bytes) -> str:
        # Replace with actual HTTP upload to your vector store
        print(f"Uploading {filename} ({len(content)} bytes) to {vector_store_id}")
        return f"file-{hash(content) % 100000}"

    async def delete_file(self, file_id: str) -> None:
        print(f"Deleting file {file_id}")

async def main():
    backend = MyVectorStoreBackend(base_url="https://my-vs.example.com", api_key="secret")
    foundry_client = FoundryChatClient(model="gpt-4o")
    file_search_tool = foundry_client.get_file_search_tool(vector_store_ids=["vs-custom"])

    fs_config = FileSearchConfig(
        backend=backend,
        vector_store_id="vs-custom",
        file_search_tool=file_search_tool,
    )
    provider = ContentUnderstandingContextProvider(
        endpoint="https://my-cu.cognitiveservices.azure.com",
        api_key="my-api-key",
        file_search_config=fs_config,
    )
    session = AgentSession()
    session.add_context_provider(provider)

    agent = Agent(client=foundry_client, instructions="Answer from documents.")
    result = await agent.run("What is the project timeline?", session=session)
    print(result.text)

asyncio.run(main())
```

---

## 8. `AgentFrameworkTracer`

**Module:** `agent_framework.lab.lightning`  
**Install:** `pip install agent-framework` (included via `agent-framework-lab-lightning` dependency)

`AgentFrameworkTracer` bridges Agent Framework OpenTelemetry instrumentation with the
Agent Lightning RL training loop. It subclasses `AgentOpsTracer` from the `agentlightning`
library and adds lifecycle hooks that toggle OTel tracing on around training runs,
avoiding span overhead in non-training code paths.

### Class hierarchy

```
agentlightning.AgentOpsTracer
  └── AgentFrameworkTracer
```

### Method reference

```
init() -> None
    # Sets OBSERVABILITY_SETTINGS.enable_otel = True, then calls super().init()
    # Enables OTel spans/traces before the training loop begins

teardown() -> None
    # Calls super().teardown(), then sets OBSERVABILITY_SETTINGS.enable_otel = False
    # Disables OTel after training to avoid overhead in non-training paths
```

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| `OBSERVABILITY_SETTINGS` | `agent_framework.observability.OBSERVABILITY_SETTINGS` — the same singleton used by `configure_otel_providers` |
| `AgentOpsTracer` | `agentlightning.AgentOpsTracer` — the library's standard tracer protocol |
| Order of operations in `init` | OTel enabled **before** `super().init()` so the training controller sees spans from the very first agent call |
| Order of operations in `teardown` | `super().teardown()` called **before** disabling OTel so in-flight spans can flush |
| Usage | Pass an instance to the Agent Lightning training harness `tracer=` parameter |

**Example 1 — attach the tracer to an Agent Lightning training run:**

```python
import asyncio
from agent_framework.lab.lightning import AgentFrameworkTracer
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient

async def main():
    tracer = AgentFrameworkTracer()

    # Simulate the Agent Lightning lifecycle
    tracer.init()   # OTel enabled; training controller can now collect traces

    client = FoundryChatClient(model="gpt-4o")
    agent = Agent(client=client, instructions="Solve the task.")

    # During training, the harness calls agent.run(...) and records OTel spans
    result = await agent.run("What is 7 * 8?")
    print(result.text)

    tracer.teardown()  # flushes in-flight spans, then disables OTel

asyncio.run(main())
```

**Example 2 — verify OTel toggles around init/teardown:**

```python
import asyncio
from agent_framework.lab.lightning import AgentFrameworkTracer
from agent_framework.observability import OBSERVABILITY_SETTINGS

async def main():
    tracer = AgentFrameworkTracer()

    print(f"Before init: enable_otel={OBSERVABILITY_SETTINGS.enable_otel}")
    tracer.init()
    print(f"After init: enable_otel={OBSERVABILITY_SETTINGS.enable_otel}")   # True
    tracer.teardown()
    print(f"After teardown: enable_otel={OBSERVABILITY_SETTINGS.enable_otel}")  # False

asyncio.run(main())
```

**Example 3 — subclass to add custom span attributes for RL reward logging:**

```python
import asyncio
from agent_framework.lab.lightning import AgentFrameworkTracer
from agent_framework.observability import OBSERVABILITY_SETTINGS

class RewardLoggingTracer(AgentFrameworkTracer):
    """Extends AgentFrameworkTracer to attach reward metadata to OTel spans."""

    def __init__(self, reward_tag: str = "default"):
        super().__init__()
        self._reward_tag = reward_tag

    def init(self) -> None:
        super().init()   # enables OTel
        # Additional setup: e.g. register a custom span processor for reward labels
        print(f"RewardLoggingTracer initialised for tag='{self._reward_tag}'")

    def teardown(self) -> None:
        print(f"RewardLoggingTracer tearing down for tag='{self._reward_tag}'")
        super().teardown()  # flushes + disables OTel

async def main():
    tracer = RewardLoggingTracer(reward_tag="airline-v1")
    tracer.init()
    print(f"OTel active: {OBSERVABILITY_SETTINGS.enable_otel}")
    tracer.teardown()
    print(f"OTel active: {OBSERVABILITY_SETTINGS.enable_otel}")

asyncio.run(main())
```

---

## 9. `TaskRunner` (lab.tau2)

**Module:** `agent_framework.lab.tau2`  
**Install:** `pip install agent-framework` (requires `tau2` package for full benchmark execution)

`TaskRunner` orchestrates the [Tau2](https://github.com/sierra-research/tau2) airline
customer service benchmark. It builds a cyclic multi-agent workflow (orchestrator →
assistant → orchestrator → user simulator → orchestrator), runs a conversation to
completion or termination, and evaluates the result using the tau2 `evaluate_simulation`
API. `patch_env_set_state` / `unpatch_env_set_state` monkey-patch the Tau2 environment
for controlled test replay.

### Constructor reference

```
TaskRunner(
    max_steps: int,
    assistant_sampling_temperature: float = 0.0,
    assistant_window_size: int = 32768,
)
```

### Constants

```python
ASSISTANT_AGENT_ID = "assistant_agent"
USER_SIMULATOR_ID  = "user_simulator"
ORCHESTRATOR_ID    = "orchestrator"
```

### Method reference

```
reinit() -> TaskRunner
    # Resets all state; returns self — enables reuse without reinstantiation

build_conversation_workflow(assistant_agent, user_simulator_agent) -> Workflow
    # Builds cyclic WorkflowBuilder: orchestrator → assistant → orchestrator → user → orchestrator
    # max_iterations=10000; termination via should_not_stop condition

should_not_stop(response) -> bool | TerminationReason
    # step_count >= max_steps → TerminationReason.MAX_STEPS
    # STOP / TRANSFER / OUT_OF_SCOPE in user text → TerminationReason.USER_STOP
    # Agent side always returns False (agent cannot signal stop)

run(task, assistant_chat_client, user_simulator_chat_client) -> tuple[list[Message], TerminationReason]
    # Builds agents + workflow; runs with greeting "Hi! How can I help you today?"
    # Assembles full conversation from: greeting + message_store.list_all_messages() + _final_user_message

evaluate(task_input, conversation, termination_reason) -> float
    # Converts to tau2 SimulationRun; calls evaluate_simulation(..., evaluation_type=EvaluationType.ALL,
    # solo_mode=False, domain="airline"); returns full_reward_info.reward (0.0 if None)
```

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| Memory | Uses `SlidingWindowChatMessageStore(system_message, tool_definitions, max_tokens=32768)` for the assistant agent |
| `conversation_orchestrator` | Flips message roles and routes to the opposite agent via `ctx.send_message(..., target_id=...)` |
| Conversation assembly | Three-part: hardcoded greeting + `message_store.list_all_messages()` + `_final_user_message` |
| `reinit()` | Returns `self` — enables `runner.reinit().run(...)` chaining |
| `patch_env_set_state` | Monkey-patches Tau2 environment `set_state` for deterministic test replay |
| `unpatch_env_set_state` | Removes the monkey-patch, restoring original `set_state` |
| Reward | 0.0 is returned when `full_reward_info.reward` is `None` |

**Example 1 — run a single Tau2 task and evaluate:**

```python
import asyncio
from agent_framework.lab.tau2 import TaskRunner, ASSISTANT_AGENT_ID
from agent_framework.foundry import FoundryChatClient

async def main():
    runner = TaskRunner(max_steps=20, assistant_sampling_temperature=0.0)

    assistant_client = FoundryChatClient(model="gpt-4o")
    user_sim_client = FoundryChatClient(model="gpt-4o-mini")

    # task is a tau2 Task object loaded from the benchmark dataset
    # For illustration, assume task is already loaded
    # conversation, reason = await runner.run(task, assistant_client, user_sim_client)
    # reward = runner.evaluate(task.input, conversation, reason)
    # print(f"Reward: {reward:.3f}, Termination: {reason}")

    print(f"Runner initialised with max_steps={runner.max_steps}")
    print(f"Assistant agent ID: {ASSISTANT_AGENT_ID}")

asyncio.run(main())
```

**Example 2 — reuse `TaskRunner` across multiple tasks with `reinit`:**

```python
import asyncio
from agent_framework.lab.tau2 import TaskRunner
from agent_framework.foundry import FoundryChatClient

async def run_benchmark(tasks: list, assistant_client, user_sim_client) -> list[float]:
    runner = TaskRunner(max_steps=15)
    rewards = []

    for task in tasks:
        runner.reinit()  # reset state; reuse the same runner object
        conversation, reason = await runner.run(task, assistant_client, user_sim_client)
        reward = runner.evaluate(task.input, conversation, reason)
        rewards.append(reward)
        print(f"Task {task.id}: reward={reward:.3f}, reason={reason}")

    return rewards

async def main():
    assistant_client = FoundryChatClient(model="gpt-4o")
    user_sim_client = FoundryChatClient(model="gpt-4o-mini")
    print("TaskRunner reuse pattern demonstrated (no tasks loaded in this example).")

asyncio.run(main())
```

**Example 3 — monkey-patch Tau2 environment for deterministic test replay:**

```python
import asyncio
from agent_framework.lab.tau2 import (
    TaskRunner,
    patch_env_set_state,
    unpatch_env_set_state,
    ASSISTANT_AGENT_ID,
    USER_SIMULATOR_ID,
    ORCHESTRATOR_ID,
)
from agent_framework.foundry import FoundryChatClient

async def main():
    # Patch the Tau2 environment to replay a fixed state sequence
    patch_env_set_state()
    print("Tau2 env.set_state patched for deterministic replay.")

    try:
        runner = TaskRunner(max_steps=5, assistant_sampling_temperature=0.0)
        assistant_client = FoundryChatClient(model="gpt-4o")
        user_sim_client = FoundryChatClient(model="gpt-4o-mini")

        # With the patch active, environment state transitions are controlled
        # rather than drawn from the live simulator.
        print(f"Agent IDs: assistant={ASSISTANT_AGENT_ID}, "
              f"user={USER_SIMULATOR_ID}, orchestrator={ORCHESTRATOR_ID}")

    finally:
        unpatch_env_set_state()   # always restore, even on failure
        print("Patch removed; original set_state restored.")

asyncio.run(main())
```

---

## 10. New `FoundryChatClient` hosted tool factories

**Module:** `agent_framework.foundry` (from `agent_framework_foundry._chat_client`)  
**Install:** `pip install agent-framework[foundry]`

Eight new factory methods were added to `FoundryChatClient` in 1.9.0, complementing the
existing `get_file_search_tool` and `get_code_interpreter_tool`. All return tool instances
accepted by `Agent(tools=[...])` or `agent.default_options["tools"]`. Validation errors
(missing required parameters) are raised at factory call time rather than at agent run
time.

### New factory methods

| Method | Purpose |
|--------|---------|
| `get_azure_ai_search_tool(index_connection_name, index_name, *, query_type=None, semantic_config_name=None, retrieval_reasoning_effort=None, top_n_documents=None)` | Hosted Azure AI Search grounding tool |
| `get_sharepoint_tool(sharepoint_connection_name, ...)` | Hosted SharePoint / OneDrive content grounding |
| `get_fabric_tool(connection_id, ...)` | Hosted Microsoft Fabric data grounding |
| `get_memory_search_tool(vector_store_id, ...)` | Hosted Foundry Memory search |
| `get_computer_use_tool(environment="browser", *, display_height=768, display_width=1024, display_number=1)` | Computer use tool for browser automation |
| `get_browser_automation_tool(connection_id, ...)` | Hosted browser automation tool |
| `get_a2a_tool(agent_url, ...)` | Agent-to-Agent (A2A) delegating tool |
| `get_mcp_tool(server_url=None, *, project_connection_id=None, server_label=None, allowed_tools=None)` | Hosted MCP server tool (no local subprocess) |

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| `get_file_search_tool` validation | Raises `ValueError` if `vector_store_ids` is missing or empty |
| `get_mcp_tool` validation | Raises `ValueError` if both `server_url` and `project_connection_id` are `None` |
| Hosted vs local MCP | `get_mcp_tool` uses a Foundry-hosted MCP server — no local subprocess, no `MCPStdioTool` overhead |
| A2A delegation | `get_a2a_tool` forwards tasks to another agent endpoint; useful for composition across independently deployed agents |
| Computer use defaults | `environment="browser"`, `display_height=768`, `display_width=1024`, `display_number=1` |
| Tool composition | All factory methods return values that compose freely in `tools=[...]` lists |

**Example 1 — grounded responses from Azure AI Search:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient

async def main():
    client = FoundryChatClient(model="gpt-4o")

    search_tool = client.get_azure_ai_search_tool(
        index_connection_name="my-search-connection",
        index_name="product-catalogue",
        query_type="semantic",
        semantic_config_name="default",
        top_n_documents=5,
    )
    agent = Agent(
        client=client,
        tools=[search_tool],
        instructions="Answer product questions using the search index.",
    )
    result = await agent.run("What wireless keyboards do you sell under $50?")
    print(result.text)

asyncio.run(main())
```

**Example 2 — SharePoint grounding with Fabric data:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient

async def main():
    client = FoundryChatClient(model="gpt-4o")

    sharepoint_tool = client.get_sharepoint_tool(
        sharepoint_connection_name="my-sharepoint-connection",
    )
    fabric_tool = client.get_fabric_tool(
        connection_id="my-fabric-connection-id",
    )
    agent = Agent(
        client=client,
        tools=[sharepoint_tool, fabric_tool],
        instructions=(
            "Answer business questions by searching SharePoint documents "
            "and querying Fabric data warehouses."
        ),
    )
    result = await agent.run(
        "What were our total sales last quarter according to the Fabric warehouse "
        "and what does the SharePoint roadmap say about next quarter?"
    )
    print(result.text)

asyncio.run(main())
```

**Example 3 — A2A delegation and hosted MCP in one agent:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient

async def main():
    client = FoundryChatClient(model="gpt-4o")

    # Delegate to a separately deployed specialist agent via A2A
    a2a_tool = client.get_a2a_tool(
        agent_url="https://my-specialist-agent.example.com/a2a",
    )

    # Hosted MCP server — no local subprocess; get_mcp_tool raises ValueError
    # if both server_url and project_connection_id are None
    hosted_mcp = client.get_mcp_tool(
        server_url="https://my-mcp.example.com/mcp",
        server_label="data-tools",
        allowed_tools=["query_database", "list_tables"],
    )

    memory_tool = client.get_memory_search_tool(
        vector_store_id="vs-memory-store",
    )

    agent = Agent(
        client=client,
        tools=[a2a_tool, hosted_mcp, memory_tool],
        instructions=(
            "Coordinate across specialist agents, external tools, and memory "
            "to answer complex business questions."
        ),
    )
    result = await agent.run(
        "Check our memory store for past decisions, query the database for "
        "current figures, and ask the specialist agent for its recommendation."
    )
    print(result.text)

asyncio.run(main())
```

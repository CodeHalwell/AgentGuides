---
title: "LangGraph Validation Report"
description: "How the LangGraph guides were checked against the installed Python and TypeScript packages, what was corrected, and what is known to be outstanding."
framework: langgraph
---

**Last validated:** %%versions-checked%%
**Python:** `langgraph` %%version=langgraph%% (with `langgraph-checkpoint` %%version=langgraph-checkpoint%%, `langgraph-prebuilt` %%version=langgraph-prebuilt%%, `langgraph-checkpoint-sqlite` %%version=langgraph-checkpoint-sqlite%%, `langgraph-checkpoint-postgres` %%version=langgraph-checkpoint-postgres%%, `langchain` %%version=langchain%%, `langchain-core` %%version=langchain-core%%)
**TypeScript:** `@langchain/langgraph` %%version=@langchain/langgraph%% (with `@langchain/core` %%version=@langchain/core%%, `langchain` %%version=langchainjs%%, `@langchain/langgraph-checkpoint` %%version=@langchain/langgraph-checkpoint%%)

This page replaces an earlier report dated November 2025. That report described a repository layout that no longer exists and recorded two changes that were themselves wrong (it replaced `create_react_agent` with LangChain 0.x's `AgentExecutor` / `create_tool_calling_agent`, which are not part of LangChain 1.x).

---

## Method

1. Installed the current packages in clean environments (a Python venv and a Node project).
2. Extracted every import from every Python and TypeScript code block in [the Python guide](/langgraph-guide/python/) and [the TypeScript guide](/langgraph-guide/typescript/) and checked that each module and name resolves.
3. Parsed the Python blocks and compared keyword arguments on the main LangGraph APIs (`add_node`, `compile`, `Send`, `Command`, the policy classes, `ToolNode`, store methods, `invoke`/`stream`, `create_agent` and others) with the installed signatures.
4. Ran representative snippets end to end — state schemas, subgraphs, `Command` routing, node caching, deferred nodes, timeouts and error handlers, streaming chunk shapes, stores, agents with a fake model, agent middleware — and rewrote examples that failed.
5. Read the release notes between the versions the pages previously named and the current ones. Neither package had a major release in that range; the TypeScript changes from 1.3.5 to 1.5.2 are additive (node timeouts and error handlers, `setNodeDefaults`, `RunControl`, `DeltaChannel`, `GraphCallbackHandler`, per-node `tracePolicy`) plus fixes.

## Principal corrections

**TypeScript**

- `@langgraphjs/toolkit` had been recommended as the new home of `createReactAgent`. That npm name is a security placeholder, not a LangGraph package. `createReactAgent` is deprecated in favour of `createAgent` from `langchain`.
- Removed APIs that do not exist: `AgentExecutor` from `@langchain/core/agents`, `createOpenAIFunctionsAgent` / `createStructuredChatAgent` from `@langchain/langgraph/prebuilt`, `ZodOutputParser`, `SerpAPITool`, `BufferMemory` / `ConversationChain` from `@langchain/core`, hand-written checkpoint savers with invented method names, and the node options `cache`, `cacheKey`, `cacheTTL` and `deferred` (the real ones are `cachePolicy` with `compile({ cache })`, and `defer`).
- `tool()` examples passed the schema first and the function second; the order is the reverse.
- `.stream()` returns a promise and must be awaited; multi-mode chunks are `[mode, payload]` tuples and `messages` chunks are `[message, metadata]` tuples.
- `Annotation({ default })` without a `reducer` silently ignores the default (and is a type error); such fields now have an explicit last-value reducer.
- `new StateGraph<T>()` without a schema, and `channels: { x: { value_type: ... } }`, replaced by `Annotation.Root`, Zod or `StateSchema`.

**Python**

- The production guide's "v1.0.3 Production Patterns" section used invented modules (`langgraph.cache.DistributedCache`, `@cache_node`, `langgraph.graph.deferred`, `langgraph.llm_hooks`); it was rewritten with `CachePolicy` + `RedisCache`, `defer=True`, `create_agent` middleware and `AsyncPostgresStore`.
- `AgentExecutor`-based supervisor and ReAct examples rewritten with `langchain.agents.create_agent`.
- Wrong import paths fixed: `StateSnapshot` (in `langgraph.types`), `RemoteGraph` (in `langgraph.pregel.remote`), `SerializerProtocol` (in `langgraph.checkpoint.serde.base`). `langgraph.types.call` is not public; the example now uses `@task`.
- `SqliteSaver` / `PostgresSaver` / `AsyncPostgresStore` `.from_conn_string(...)` are context managers; several examples assigned them directly.
- `store.search` / `asearch` take the namespace prefix positionally; keyword use raised `TypeError`.
- `langgraph[postgres]` is not an extra; PostgreSQL support is `langgraph-checkpoint-postgres` with psycopg 3, not `psycopg2`.
- `set_node_defaults` has no `trace_policy` parameter, and `add_conditional_edges` no longer has `then=`.
- Retired model IDs replaced with current ones.

## Known issues and caveats

- **Python `error_handler` in parallel supersteps.** With `langgraph` %%version=langgraph%%, a node-level `error_handler` handled a failure when the node ran on its own, but not when the failing task ran in parallel with other tasks in the same superstep (for example `Send` fan-out); the exception propagated and failed the run. The production guide therefore catches errors inside fan-out workers.
- **Stale deprecation pointer.** `langgraph.prebuilt.interrupt.HumanInterrupt` and related classes warn that they moved to `langchain.agents.interrupt`, but that module does not exist in `langchain` 1.4.x. The guides point to `HumanInTheLoopMiddleware` instead.
- **Coverage.** Imports and keyword arguments were checked for every block; full execution was limited to representative snippets, since many examples need model API keys or external services. Signature dumps in the reference pages were spot-checked (for example `RetryPolicy`, `TimeoutPolicy`, `ExecutionInfo`, `ErrorCode`, `DEFAULT_RECURSION_LIMIT`), not exhaustively.
- **Overlap.** The Python recipes include three separate `ToolRuntime` recipes (12, 17 and 31) that cover similar ground with different examples.

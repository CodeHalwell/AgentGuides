# Agent Guides: end-to-end assessment, 21 October 2026

Fourteen read-only assessors, one per framework, checked every page against the current release (versions from `src/data/versions.json`): every code block extracted and parsed, type-checked or compiled, quick-starts run against each framework's mock or test model, prose spot-checked against source and changelogs.

## Scorecard (1–5)

| Guide | Code | Currency | Accuracy | Coverage | Structure | Consistency | Production | Mean | Blockers |
|---|---|---|---|---|---|---|---|---|---|
| AG2 1.x | 5 | 3 | 4 | 3 | 3 | 4 | 3 | 3.6 | 0 |
| Mistral | 4 | 4 | 3 | 3 | 3 | 4 | 3 | 3.4 | 2 |
| SmolAgents | 4 | 3 | 3 | 3 | 4 | 3 | 4 | 3.4 | 4 |
| Claude Agent SDK | 3 | 4 | 3 | 4 | 4 | 3 | 3 | 3.4 | 6 |
| Haystack | 4 | 3 | 3 | 3 | 3 | 4 | 3 | 3.3 | 4 |
| Semantic Kernel | 3 | 4 | 3 | 3 | 3 | 3 | 3 | 3.1 | 5 |
| LlamaIndex | 3 | 3 | 4 | 3 | 3 | 3 | 2 | 3.0 | 5 |
| Bedrock / AgentCore / Strands | 3 | 4 | 3 | 3 | 2 | 3 | 2 | 2.9 | 8 |
| Microsoft Agent Framework | 3 | 2 | 3 | 3 | 2 | 3 | 3 | 2.7 | 16 |
| AutoGen (Microsoft + AG2 Classic) | 3 | 2 | 3 | 3 | 2 | 3 | 2 | 2.6 | 7 |
| OpenAI Agents SDK | 3 | 3 | 2 | 2 | 3 | 3 | 2 | 2.6 | 10 |
| PydanticAI | 3 | 2 | 3 | 3 | 2 | 3 | 2 | 2.6 | 13 |
| Google ADK | 3 | 2 | 3 | 2 | 2 | 3 | 2 | 2.4 | 8 |
| LangGraph | 2 | 2 | 2 | 3 | 2 | 2 | 2 | 2.1 | 26 |
| CrewAI | 2 | 2 | 2 | 3 | 2 | 2 | 1 | 2.0 | 21 |

135 blockers in total. Already fixed since the assessment: the 109 relative links that 404'd in the built site (counted as blockers by the MAF and LangGraph assessors).

## Site-wide problems (fix once, not fourteen times)

1. **Stale model IDs.** `gpt-4o`/`gpt-4o-mini` in the hundreds, `gemini-2.5-*` 327 times in ADK alone, `gpt-4-turbo`, `claude-sonnet-4-5`/`3.x`, `gpt-4.1-mini`. Version numbers are now single-sourced; model IDs aren't.
2. **Sidebar.** In most guides only the index is in the sidebar; dozens of topic pages are reachable only from index cards.
3. **Frontmatter.** `language:` missing on most topic pages; `framework:` values differ within a guide.
4. **Production pages.** In roughly eight guides they are generic FastAPI/Kubernetes scaffolding that compiles but doesn't work when assembled (duplicate routes, undefined helpers, health checks that always pass), and they miss each framework's own production features.
5. **Bloat and duplication.** 6,000-line comprehensive pages that repeat the chapter pages (LangGraph, PydanticAI), thousands of lines of private-internals reference (ADK, PydanticAI), the same topic written up in 3–8 places.
6. **Invented content.** Made-up benchmark numbers and architecture on diagrams pages, invented features (e.g. traces feeding fine-tuning), placeholder stubs ("Continuing in next part…").

## Individual reports


---

### LangGraph guide assessment (Python + TypeScript)

Assessed 2026-10-21 against `src/data/versions.json`: `langgraph` 1.2.16, `langgraph-prebuilt` 1.1.1, `langgraph-checkpoint` 4.3.0, `langchain` 1.4.6, `langchain-core` 1.6.12 (venv); `@langchain/langgraph` 1.5.2, `@langchain/core` 1.2.20, `langchain` 1.5.18, TypeScript 5.9.3. Scope: `src/content/docs/langgraph-guide/` (index, validation report, 35 Python pages, 8 TypeScript pages; about 42,000 lines). Read-only; scratch work is in `scratchpad/assess-langgraph/`.

#### Scorecard

| # | Criterion | Score | One-line justification |
|---|---|---|---|
| 1 | Correctness of code | 2 / 5 | Index quick-starts run in both languages, and most self-contained Python blocks run with a stub model. But 32 Python blocks fail on the current release (wrong error-handler signature, sync nodes with timeouts, `ToolNode.invoke` outside a graph, async `@task` misuse), and 36 of 117 TypeScript blocks fail `tsc --strict`. |
| 2 | Currency | 2 / 5 | Version placeholders are clean. But the deprecated `create_react_agent` is still the main agent API in Python, 34 OpenAI model references are to the deprecated `gpt-4o`/`gpt-4o-mini`, there are 7 retired or invalid Claude IDs, the TS guide installs the deprecated `@langchain/community`, and the pages still say "LangGraph Cloud". |
| 3 | Accuracy of prose | 2 / 5 | Several confident behavioural claims are false: the default recursion limit (says 25, is 10007), the `ToolNode` error-handling default, interrupt gotchas, InMemoryStore TTL, `destinations=` being "diagram-only", and an Errata section saying that `CachePolicy` caching and deferred nodes don't exist. The production page has invented benchmarks. |
| 4 | Coverage | 3 / 5 | Python goes very deep on core primitives (channels, policies, v3 streaming). But `langchain.agents.create_agent` and its middleware, the current agent path, are thin. TypeScript has no interrupts/HITL, `Send`, `Command`, Functional API, subgraphs or time travel. |
| 5 | Structure and navigation | 2 / 5 | The 6,000-line "comprehensive" page repeats chapters 1–10, and the retry/timeout/cache policies are documented in eight or more places. Two pages are orphans, seven TS pages have no sidebar entry, and 4 relative links break on non-index pages (the link checker misses them). |
| 6 | Consistency | 2 / 5 | Pages contradict each other: recursion limit 25 vs 10007; 7 vs 8 stream modes; timeout errors raised at `add_node` vs `compile`; `Durability` on `invoke` vs `compile`; "use middleware" (index) vs "use hooks, not a middleware layer" (ch10). Release dates for v3 streaming and `Send(timeout=)` also disagree. |
| 7 | Production readiness | 2 / 5 | Docker/K8s/health-check material is generic. It uses a non-existent `langgraph auth login`, has invented latency/throughput numbers and hard-coded prices, and has no LangSmith Deployment coverage. Testing advice is generic, with no fake-model or eval guidance. Python `RetryPolicy`/`TimeoutPolicy`/`set_node_defaults` coverage is genuinely good. |

**Verdict.** The Python reference material on LangGraph's own primitives is unusually deep and mostly source-accurate, and both quick-starts work. But the guide still teaches the deprecated `create_react_agent`/`pre_model_hook` path as the main agent API. It carries a long tail of examples that fail on the current release and a handful of false behavioural claims that would mislead a newcomer (recursion limit, `ToolNode` error handling, interrupts). It needs a consolidation pass (one canonical page per topic), a `create_agent`/middleware rewrite of ch2/ch8, and an executed-code CI check, not more pages.

#### Code verification

Method:
- **Python.** I extracted all 870 fenced `python` blocks and ran each one in the venv with stub API keys. A `sitecustomize` patches `BaseChatModel._generate_with_cache` to return a fixed `AIMessage`, so model calls succeed offline.
- **Python fragments.** Blocks that failed only with `NameError` were re-run notebook-style: common imports plus up to 10 preceding blocks from the same page, run first.
- **TypeScript.** I extracted all 119 TS/JS blocks and ran `tsc --strict` (ES2022, bundler resolution, `@types/node`). A second pass added missing standard imports (`Annotation`, `StateGraph`, `z`, …) so that real API/type errors could be told apart from fragments. Every block that compiled cleanly was then run with `tsx`.
- **Static checks.** Imports and keyword arguments on the main APIs were checked statically (all resolve, apart from expected third-party modules).

Status key: **pass** = ran to completion standalone; **pass-ctx** = ran once preceding page blocks were in scope; **fragment** = continuation snippet that still needs undefined names; **non-code** = signature listing or pseudo-literal inside a `python` fence (not valid Python); **ext** = needs an external service or optional package (Postgres, Redis, FastAPI, a LangSmith deployment, IPython…); **fail** = genuine error on the current release (each one is listed under Issues).

**Python (executed, stub chat model)**

| Page | Blocks | pass | pass-ctx | fragment | non-code | ext | fail |
|---|---|---|---|---|---|---|---|
| python/chapter-01-setup-and-core-concepts.md | 31 | 17 | 6 | 6 | 2 | 0 | 0 |
| python/chapter-02-simple-agents.md | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| python/chapter-03-multi-agent.md | 7 | 5 | 1 | 0 | 0 | 0 | 1 |
| python/chapter-04-tools.md | 12 | 8 | 2 | 2 | 0 | 0 | 0 |
| python/chapter-05-memory.md | 15 | 7 | 3 | 2 | 2 | 0 | 1 |
| python/chapter-06-streaming-and-debugging.md | 31 | 6 | 18 | 3 | 3 | 1 | 0 |
| python/chapter-07-human-in-the-loop.md | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| python/chapter-08-middleware-hooks.md | 17 | 13 | 4 | 0 | 0 | 0 | 0 |
| python/chapter-09-advanced-patterns.md | 28 | 19 | 0 | 5 | 0 | 0 | 4 |
| python/chapter-10-production.md | 23 | 6 | 7 | 9 | 0 | 1 | 0 |
| python/index.mdx | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| python/langgraph_advanced_error_recovery.md | 24 | 9 | 3 | 10 | 0 | 0 | 2 |
| python/langgraph_advanced_implementations.md | 8 | 1 | 3 | 0 | 1 | 1 | 2 |
| python/langgraph_comprehensive_guide.md | 193 | 68 | 19 | 63 | 19 | 15 | 9 |
| python/langgraph_diagrams.md | 3 | 0 | 2 | 1 | 0 | 0 | 0 |
| python/langgraph_observability_python.md | 12 | 2 | 5 | 5 | 0 | 0 | 0 |
| python/langgraph_performance_optimization.md | 21 | 6 | 5 | 9 | 1 | 0 | 0 |
| python/langgraph_production_guide.md | 14 | 4 | 1 | 3 | 0 | 6 | 0 |
| python/langgraph_recipes.md | 35 | 21 | 3 | 5 | 1 | 0 | 5 |
| python/langgraph_streaming_server_fastapi.md | 6 | 0 | 3 | 0 | 0 | 3 | 0 |
| python/reference-cache-backends.md | 16 | 7 | 5 | 3 | 0 | 1 | 0 |
| python/reference-channels.md | 38 | 28 | 1 | 2 | 3 | 0 | 4 |
| python/reference-checkpointers.md | 40 | 11 | 4 | 19 | 5 | 1 | 0 |
| python/reference-command-and-send.md | 22 | 8 | 5 | 8 | 0 | 0 | 1 |
| python/reference-functional-api.md | 22 | 8 | 9 | 3 | 2 | 0 | 0 |
| python/reference-graceful-shutdown.md | 10 | 8 | 1 | 0 | 0 | 1 | 0 |
| python/reference-managed-values.md | 11 | 5 | 0 | 5 | 1 | 0 | 0 |
| python/reference-prebuilt-nodes.md | 41 | 19 | 6 | 11 | 4 | 1 | 0 |
| python/reference-remote-graph.md | 20 | 5 | 1 | 4 | 3 | 7 | 0 |
| python/reference-runtime-and-managed-values.md | 25 | 8 | 7 | 8 | 1 | 0 | 1 |
| python/reference-state-graph.md | 37 | 8 | 1 | 23 | 4 | 0 | 1 |
| python/reference-store.md | 34 | 5 | 4 | 8 | 12 | 4 | 1 |
| python/reference-streaming-modes.md | 48 | 8 | 10 | 17 | 13 | 0 | 0 |
| python/reference-trace-policy.md | 12 | 5 | 0 | 6 | 1 | 0 | 0 |
| **Total** | **870** | **339** | **139** | **240** | **78** | **42** | **32** |

About 55% of Python blocks run as written or in page context (339 + 139 = 478 of 870). "fragment" includes 48 blocks that failed only because they extend a builder or config from an earlier block; I did not count those as failures. The 32 failures are all listed under B1–B26. Two runtime failures were artefacts of the test setup and were reclassified as ext: production_guide:882 (needs a running server) and reference-prebuilt-nodes:1285 (structured output against the stub model). Static checks: every `langgraph`/`langchain` import resolves; no wrong keyword arguments on `add_node`, `compile`, `Send`, `Command`, the policies, `ToolNode`, the store methods or `invoke`/`stream`.

**TypeScript (`tsc --strict`, then `tsx` for clean blocks)**

| Page | Blocks | Clean | Fragment only (missing imports/names) | Real strict/type errors | Runtime failures among clean |
|---|---|---|---|---|---|
| typescript/index.mdx | 1 | 1 | 0 | 0 | 0 (ran: prints echo reply) |
| typescript/langchain_langgraph_comprehensive_guide.md | 84 | 36 | 22 | 26 | 1 (L2210) + 2 need real OpenAI key + 1 needs Redis |
| typescript/langchain_langgraph_recipes.md | 13 | 6 | 0 | 7 | 0 |
| typescript/langchain_langgraph_production_guide.md | 13 | 8 | 2 | 3 | 0 (1 server, timed out as expected) |
| typescript/langgraph_middleware_typescript.md | 3 | 2 | 1 | 0 | 0 |
| typescript/langgraph_observability_typescript.md | 2 | 1 | 1 | 0 | 0 |
| typescript/langgraph_streaming_server_express.md | 1 | 1 | 0 | 0 | 0 (server) |
| python pages (FastAPI JS client, ch6 TSX sketch) | 2 | 1 | 0 | 1 (pseudo-code) | 0 |
| **Total** | **119** | **56** | **26** | **37** | **1** |

Of the 26 TS comprehensive-guide blocks and 7 recipe blocks with real errors, 24 share one cause: the builder is used statement by statement (`const g = new StateGraph(S); g.addNode(...); g.addEdge(START, 'x')`). LangGraph.js tracks node names in the builder's type, so under `strict` every `addEdge` gets TS2345 (`'"x"' is not assignable to '"__start__" | "__end__"'`). The code runs, but it does not type-check. The rest are implicit `any` (TS7006/7034), `this` before `super()`, `config.writer` on a tool config type, `js-tiktoken`'s `encoding_for_model` (actual name `encodingForModel`), and `chunk.content` passed to `process.stdout.write` (`string | ContentBlock[]`).

Also, the guide's own recommended `tsconfig.json` (typescript/langchain_langgraph_comprehensive_guide.md:60–80, `"lib": ["ES2020"]`, `"moduleResolution": "node"`) does not compile the TS index example: `messages.at(-1)` gives TS2550.

Checks that pass:
- `node scripts/check-versions.mjs | grep langgraph`: clean. The 113 placeholders all resolve.
- `node scripts/check-links.mjs`: 0 broken. That result is misleading; see B21.
- Anchor check (my own, with github-slugger): all 95 `#anchor` links resolve.
- No `deep dive` / `Vol. N` pages and no TODO stubs. No page body starts with an H1.

#### Issues

### Blocker (wrong or broken code, false claims, broken pages)

| # | Location | Problem | Fix |
|---|---|---|---|
| B1 | python/chapter-04-tools.md:85, :107, :642; python/reference-prebuilt-nodes.md:91 | Says the `ToolNode` default is `handle_tool_errors=True` ("catch all exceptions"). In langgraph-prebuilt 1.1.1 the default is `_default_handle_tool_errors`, which returns a message only for `ToolInvocationError` and **re-raises everything else**. Verified: `ToolNode([risky_lookup])` inside a graph raises `ValueError`. | Document the default as "only argument-validation errors are caught; tool exceptions propagate", and show `handle_tool_errors=True` as the opt-in catch-all. |
| B2 | python/langgraph_advanced_error_recovery.md:261, :296, :456, :468; python/langgraph_performance_optimization.md:374; python/reference-state-graph.md:339 | Error handlers declared as `(state, exception: Exception)` or `(state, error: Exception)`. LangGraph injects the failure by the `NodeError` type annotation, so every one of them fails with `TypeError: missing 1 required positional argument`. Section 3 of the error-recovery page is unusable as written. The same page (:516 ff.) correctly documents `NodeError`. | Change each one to `error: NodeError` and read `error.error` / `error.node`. |
| B3 | python/chapter-10-production.md:345, :382; python/reference-managed-values.md:117, :119; python/reference-runtime-and-managed-values.md:492 | Default `recursion_limit` stated as **25**. In 1.2.16 `DEFAULT_RECURSION_LIMIT = 10007` (env `LANGGRAPH_DEFAULT_RECURSION_LIMIT`), as error_recovery:686 and comprehensive:5248 correctly say. The runtime-values example at reference-runtime-and-managed-values.md:435 loops thousands of times (it timed out after printing "Steps remaining: 4159"). A reader relying on 25 as a runaway-loop guard gets 10,007 model calls. | Correct to 10007 everywhere, recommend setting `recursion_limit` explicitly, and set it in the :435 example. |
| B4 | python/chapter-09-advanced-patterns.md:817 (node at :862); python/langgraph_comprehensive_guide.md:5208, :5931; python/reference-command-and-send.md:428; python/reference-state-graph.md:326 | `Send(..., timeout=...)` or `set_node_defaults(timeout=10.0)` used with **sync** nodes. Compile raises `ValueError: Node timeouts are only supported for async nodes`. | Make the target nodes `async def` (and use `asyncio.sleep`). |
| B5 | python/chapter-03-multi-agent.md:651 (rule at :709, claims at :378, :719) | Child node declares `destinations={"escalation_handler"}`, a node that exists only in the parent, so `child_builder.compile()` raises `Found edge ending at unknown node`. The text says `destinations=` is "diagram-only" and "does not change execution", but it adds edges that are validated at compile time. | Drop `destinations` on the child (or annotate `Command[Literal[...]]` in the parent). Reword :378/:719: "adds graph edges for rendering; targets must exist in the same graph." |
| B6 | python/chapter-07-human-in-the-loop.md:359, :360 | (a) "Without a checkpointer, `interrupt()` raises `GraphInterrupt` uncaught". Verified false: `invoke` returns with an `__interrupt__` key, but cannot be resumed. (b) "`state.next` shows `('__interrupt__',)`". It shows the paused node, e.g. `('ask',)`. | Rewrite both gotchas to match observed behaviour. |
| B7 | python/chapter-09-advanced-patterns.md:1304–1328 | An async `@entrypoint` calls `.result()` on async task futures, which raises `asyncio.InvalidStateError: Result is not set`. The comment "asyncio.gather is incompatible" is wrong. | `return await asyncio.gather(*futures)` (or `[await f for f in futures]`). |
| B8 | python/langgraph_comprehensive_guide.md:2242 | A sync `@entrypoint` calls an `async @task`, which raises `RuntimeError: In an sync context async tasks cannot be called`. | Make the task sync, or make the entrypoint async and await. |
| B9 | python/langgraph_comprehensive_guide.md:2831 (first example of the Class & API Reference); python/langgraph_recipes.md:2129 | Graph compiled with `InMemorySaver()` but invoked without `thread_id`, which raises `ValueError: Checkpointer requires ... thread_id`. | Pass `{"configurable": {"thread_id": ...}}`. |
| B10 | python/langgraph_comprehensive_guide.md:4500, :4558, :4611 | `ToolNode.invoke({...})` called outside a graph raises `ValueError: Missing required config key 'N/A' for 'tools'` on the current release. These are labelled as runnable reference examples. | Wrap in a one-node `StateGraph(MessagesState)` or call the tool directly. |
| B11 | python/langgraph_recipes.md:1609–1623 (Recipe 9) | `graph.invoke(input, cfg, {"recursion_limit": 50})` raises `TypeError: Pregel.invoke() takes from 2 to 3 positional arguments`. | Merge into `cfg = {"configurable": {...}, "recursion_limit": 50}`. |
| B12 | python/langgraph_recipes.md:1298 (Recipe 8), :1870 (Recipe 12) | `create_react_agent(state_schema=...)` with a schema lacking `remaining_steps` raises `ValueError: Missing required key(s) {'remaining_steps'}`. | Rewrite on `langchain.agents.create_agent` (see M1), or add `remaining_steps: RemainingSteps`. |
| B13 | python/reference-channels.md:348, :439, :695, :742; python/langgraph_recipes.md:2651 | Channel examples fail at runtime. `EphemeralValue` is absent after clearing, so `state["flag"]` gives `KeyError` (the text says it is `None`). The `NamedBarrierValue` examples raise `InvalidUpdateError: Value None not in {...}`. | Use `state.get("flag")`, and fix the barrier examples so that only named writers write and the input does not seed the barrier key. Re-run each one. |
| B14 | python/reference-store.md:584 | `@tool` function without a docstring raises `ValueError: Function must have a docstring`. | Add docstrings. |
| B15 | python/chapter-09-advanced-patterns.md:1084 | `stream(..., version="v2")` yields `StreamPart` dicts, so `chunk.value` raises `AttributeError` (ch6 says correctly to use `part["type"]`/`part["data"]`). | `print(chunk["data"], chunk.get("interrupts"))`. |
| B16 | python/chapter-05-memory.md:522–528, :532 | "With `InMemoryStore`, pass `ttl=` per put" and "`supports_ttl` True for InMemoryStore". Verified: `supports_ttl` is `False` and `put(..., ttl=60)` raises `NotImplementedError`. "`PostgresStore` ... `abackground_tasks()`" does not exist (the real methods are `start_ttl_sweeper()` / `sweep_ttl()`). | Rewrite the TTL section around `PostgresStore(ttl=TTLConfig(...))` + `start_ttl_sweeper()`, and say InMemoryStore has no TTL. |
| B17 | python/chapter-10-production.md:469, :487 | "The installed source does not carry a runtime decorator": false, `create_react_agent` is decorated `@deprecated(..., category=LangGraphDeprecatedSinceV10)` and warns on use. ":487 use pre_model_hook ... not a middleware layer" contradicts the index's guidance to use `create_agent` middleware. | Correct both, and point to `create_agent` + `AgentMiddleware`. |
| B18 | python/chapter-10-production.md:339, :722 | "Choose `"sync"` ... (required for `interrupt()`)": false (interrupt/resume verified under `"async"` and `"exit"`). The quick-reference table says `compile(..., durability="async")`, but `compile()` has no `durability` parameter (and :314 says so itself). | Drop the "required" claim and change the table to `invoke(..., durability="async")`. |
| B19 | python/langgraph_production_guide.md:320–324; index.mdx:50 | `langgraph auth login` is not a command (langgraph-cli 0.4.37 has `build, deploy [beta], dev, dockerfile, new, up, validate`). "LangGraph Cloud" is now LangSmith Deployment. | Replace with `langgraph deploy` and a link to the LangSmith Deployment docs, and rename throughout (also reference-runtime-and-managed-values.md:294–346, langgraph_diagrams.md:404). |
| B20 | python/langgraph_advanced_implementations.md:22–80, :244–257, :340–410 (orphan page) | The page claims "complete, runnable" examples, but:<br>- The router maps conditional-edge targets to compiled graph objects and has no `START` edge, so it raises `ValueError`.<br>- The HITL example calls `interrupt(metadata=...)` (`TypeError`), resumes by sending new input instead of `Command(resume=...)`, and uses `SqliteSaver` without an import.<br>- The summariser claims to "replace the long history" but appends through `add_messages`.<br>- It uses "GPT-4 or Claude 3 Opus" (:226) and says LangGraph has no middleware (:419). | Delete the page and keep any useful prose in recipes/production (it is linked only from `src/content/docs/guides.md`). |
| B21 | python/reference-checkpointers.md:809; python/reference-functional-api.md:267; python/reference-prebuilt-nodes.md:893; python/reference-state-graph.md:420 | Relative links `./reference-…/` on non-index pages resolve in the browser to `/langgraph-guide/python/reference-state-graph/reference-streaming-modes/`, which is a 404. `scripts/check-links.mjs` resolves relative links against the file's directory, so it reports 0 broken. | Use root-absolute links (`/langgraph-guide/python/reference-streaming-modes/`), per AGENTS.md. Also consider making check-links resolve relative links against the page URL. |
| B22 | typescript/langchain_langgraph_comprehensive_guide.md:2210–2250 | Adds node `'process'` twice ("traditional" vs "new .addNode() approach", which is the same method), so it throws `Error: Node 'process' already present` at runtime. | Show one builder chain, and drop the false before/after framing. |
| B23 | typescript/langchain_langgraph_comprehensive_guide.md:1305–1320 | `DataProcessingTool` passes `this.execute.bind(this)` to `super()` before `this` exists (TS17009; ReferenceError at runtime). | Use `tool()` with a closure, or pass a function that does not reference `this`. |
| B24 | python/langgraph_comprehensive_guide.md:2220–2235 (and banner :12) | The "Errata" section says node caching does not exist ("use Store or `functools.lru_cache`") and that "Deferred Nodes ... are not real". `CachePolicy` + `compile(cache=)` and `add_node(defer=True)` are real and documented elsewhere in the same page (:2523, :2551). It also says "installed `langgraph 1.2.0`". | Delete the Errata section and banner (history belongs in git). Where needed, point to the real `CachePolicy` / `defer=True` sections. |
| B25 | python/reference-state-graph.md:278 | "the string `"END"` or `END` → stop". `END == "__end__"`, and returning `"END"` routes to a non-existent node. | "return `END` (the string `"__end__"`)". |
| B26 | python/chapter-09-advanced-patterns.md:143, :206; python/langgraph_comprehensive_guide.md:1845, :1908 (Tree-of-Thoughts); python/langgraph_comprehensive_guide.md:656 (parallel workers) | A function returning `list[Send]` is registered as a **node** (`add_node("generate", generate_thoughts)`) and sends to `"explore_thought"`, but the node is named `"explore"`. Run result: `InvalidUpdateError: Expected dict, got [Send(...)]`. | Return `Send`s from a routing function in `add_conditional_edges("generate", fan_out)` (or `START`), and match the node names. |

### Major (missing core feature, stale model IDs, confusing structure)

| # | Location | Problem | Fix |
|---|---|---|---|
| M1 | python/chapter-02-simple-agents.md:192 (Example 4); python/chapter-08-middleware-hooks.md (whole page, from :35); python/langgraph_recipes.md:1294; python/reference-prebuilt-nodes.md (6 uses) | The deprecated `create_react_agent` (it warns `LangGraphDeprecatedSinceV10`) is the primary agent API: 22 call sites vs 4 for `langchain.agents.create_agent`. Ch2 never mentions the deprecation. Ch8 "Middleware" is about `pre_model_hook`/`post_model_hook`, not `AgentMiddleware`, and the built-ins (`HumanInTheLoopMiddleware`, `SummarizationMiddleware`, `ModelCallLimitMiddleware`, PII…) are barely shown. | Rewrite ch2 Example 4 and ch8 on `create_agent(model, tools, middleware=[...])`. Keep one short "legacy `create_react_agent`" section. |
| M2 | Stale model IDs (see the list below) | 34 references to `gpt-4o`/`gpt-4o-mini`, which are deprecated with shutdown on 2027-01-14 per OpenAI's deprecations page (replacement `gpt-6-luna`; current flagships `gpt-6-astra`, `gpt-6.1-sol`). There are also 7 retired or invalid Claude IDs and 2 obsolete prose names. | Use `claude-sonnet-5-5` / `claude-haiku-5-5` / `claude-opus-5-5` and current OpenAI IDs, or read the model from an env var. |
| M3 | ~24 TS blocks (comprehensive guide :253, :519, :1759, :1924, :2067, :2314, :2380, :2503, :2559, :2741, :2838, :3049, :3143, :3245, :3361, :3703, :4199; recipes :143, :239, :588, :709, :816, :945, :1052) | The builder is used statement by statement, so these blocks fail `tsc --strict` (TS2345 on every `addEdge`). The very first "Zero → Hero" TS example (:253) is one of them. | Chain `.addNode().addEdge()...compile()` (as the index example does). |
| M4 | typescript/langchain_langgraph_comprehensive_guide.md:60–80 | The recommended `tsconfig` (`lib: ["ES2020"]`, `moduleResolution: "node"`, `ts-node`) breaks the guide's own index example (`Array.prototype.at`, TS2550). | `target/lib: ES2022`, `module: "NodeNext"`/`moduleResolution: "NodeNext"` (or `bundler`), and `tsx` instead of `ts-node`. |
| M5 | TypeScript guide overall; index.mdx:31–52 | TS has no interrupts/HITL (one passing mention), and no `Send`, `Command`, Functional API (`entrypoint`/`task`), subgraphs, `getStateHistory`/time travel, Store-in-tools or `RemoteGraph`. The root index claims "What you'll find in both guides: … Interrupts, multi-stage approvals, time-travel debugging". | Add TS sections for interrupt/`Command({resume})`, `Send` fan-out, subgraphs and the Functional API, or narrow the index claim. |
| M6 | typescript/langchain_langgraph_comprehensive_guide.md:98, :1489–1498 | Recommends `npm install @langchain/community` and `SerpAPI`/`Calculator` from it. npm marks `@langchain/community@1.1.29` **deprecated** (see langchainjs-community#61). | Remove it, or use `tool()` wrappers / provider packages. Mention the deprecation. |
| M7 | python/langgraph_comprehensive_guide.md (6,045 lines) | Repeats chapters 1–10, then adds a 3,300-line "Class & API Reference" that overlaps the 14 reference pages. `RetryPolicy`/`TimeoutPolicy`/`CachePolicy`/`set_node_defaults` are documented in ch1, ch8, ch9, ch10, error-recovery, performance, reference-state-graph, recipes 22/30 and the comprehensive page. reference-managed-values.md duplicates reference-runtime-and-managed-values.md:410–770. Recipes 12/17/31 (`ToolRuntime`) and 11/19/23 (`BinaryOperatorAggregate`) overlap. | Make the reference pages canonical and the chapters tutorial-only. Delete the "comprehensive" page (or generate it), merge managed-values into one page, and merge duplicate recipes. |
| M8 | astro.config.mjs:179; langgraph_validation_report.md; python/langgraph_advanced_implementations.md | The TS sidebar is a single link, so 7 TS pages have no sidebar entries. The validation report is linked nowhere (orphan), and advanced_implementations is not in the sidebar or index. | Add a TS sidebar group, link the validation report from the index (or drop it), and delete advanced_implementations (B20). |
| M9 | python/chapter-06-streaming-and-debugging.md:495; python/chapter-09-advanced-patterns.md:815; python/chapter-10-production.md:442; python/chapter-04-tools.md:543; python/recipes headings (e.g. :1141 "(v1.2.1)", :3123 "(v1.2.11)") | Release history is wrong. v3 `stream_events` and `TimeoutPolicy`/`Send(timeout=)` first ship in **1.2.0** (checked against the 1.1.10/1.2.0/1.2.4/1.2.11 wheels), not 1.2.11 / 1.2.1. `ToolCallRequest.override()` is in langgraph-prebuilt ≥1.0.2, not "1.2.4". Per-release tags in recipe headings date the page. | Correct the versions and remove version tags from headings. |
| M10 | python/chapter-06-streaming-and-debugging.md:21–23 vs python/reference-streaming-modes.md:65, :77 | Ch6 says 7 modes (no `"tools"`). The reference says 8 and prints a `StreamMode` Literal that includes `"tools"`, but the installed `langgraph.types.StreamMode` has 7 values (`"tools"` is accepted at runtime only). | Pick one story: "seven typed modes plus `"tools"` (accepted, not in the Literal)". |
| M11 | python/chapter-08-middleware-hooks.md:747 vs python/chapter-10-production.md:108, :135, :275 | Ch8 says sync nodes with `timeout` raise "at node registration time (`add_node`)". Verified: `add_node` accepts it and `compile()` raises. | Fix ch8. |
| M12 | python/chapter-10-production.md:23 | Says to attach the policy with `add_node(..., retry=...)`. `retry` is deprecated (`LangGraphDeprecatedSinceV05`). | `retry_policy=`. |
| M13 | python/langgraph_production_guide.md:853–878, :1021–1029, :1057 | Invented "Expected Metrics" (P95 100–200 ms for "simple chat", 20–50 req/s per instance, 50–200 MB per execution) with no source. Hard-coded prices (`claude-sonnet-5-5` $2/$10 per M tokens, `gpt-4o`) are unverified. "Fall back to GPT-4 Mini" names no real model. | Delete the benchmark block or label it as an example to measure yourself. Link the provider pricing pages instead of hard-coding rates. |
| M14 | python/langgraph_production_guide.md:43–53; python/langgraph_advanced_implementations.md:501ff; TS production guide (no testing section) | Testing guidance is a generic checklist. Nothing on `GenericFakeChatModel`/`FakeMessagesListChatModel` (Python) or `FakeListChatModel` (TS), on testing nodes with `InMemorySaver`, interrupt/resume tests or LangSmith evaluations. | Add a short, runnable testing section per language. |
| M15 | typescript/langchain_langgraph_diagrams.md:889–920, :33–36 | The "TypeScript Type Hierarchy for Agents" (`Agent<In,Out>`, `SimpleAgent`, `StructuredAgent`, `MultiAgentOrchestrator`) is invented. "Buffer/Window/Summary Memory" are listed as part of `@langchain/core`; they are legacy classes not in core. | Remove the invented hierarchy and the memory box. Draw real types (`StateGraph`, `CompiledStateGraph`, `ReactAgent` from `createAgent`). |

**Stale model IDs (M2)**:
- **Claude:**
  - `claude-opus-4-7`: python/chapter-09-advanced-patterns.md:80; python/reference-functional-api.md:569
  - `claude-haiku-4-5`: python/reference-functional-api.md:569
  - `claude-opus-4-8`: python/langgraph_recipes.md:2367, :2492
  - Invalid bare `claude-opus` / `claude-haiku`: python/chapter-09-advanced-patterns.md:695
  - Prose "Claude 3 Opus" and "GPT-4": python/langgraph_advanced_implementations.md:226
- **OpenAI `gpt-4o-mini`:**
  - python/reference-prebuilt-nodes.md:33, :378, :456, :518, :569, :937, :996, :1175, :1259, :1304
  - python/reference-streaming-modes.md:362, :379, :750
  - python/langgraph_recipes.md:3271
  - python/chapter-03-multi-agent.md:98, :99
  - python/reference-runtime-and-managed-values.md:373, :381
  - typescript/langgraph_middleware_typescript.md:32
  - typescript/langchain_langgraph_comprehensive_guide.md:639, :642, :3547
  - typescript/langchain_langgraph_production_guide.md:369
- **OpenAI `gpt-4o`:**
  - python/reference-prebuilt-nodes.md:1259
  - python/langgraph_production_guide.md:1025
  - python/reference-runtime-and-managed-values.md:372
  - typescript/langchain_langgraph_comprehensive_guide.md:116, :142, :321, :3689
  - typescript/langchain_langgraph_production_guide.md:122, :339, :368, :400
- **Invented name:** "GPT-4 Mini" at python/langgraph_production_guide.md:1057.
- **Status not confirmed:** `text-embedding-ada-002` in the dimension table at python/reference-store.md:331. OpenAI's deprecations page does not list it, so this needs checking.

`claude-sonnet-5-5` (45 uses) and `claude-haiku-5-5` are current.

### Minor (style, wording)

| # | Location | Problem | Fix |
|---|---|---|---|
| m1 | 78 Python blocks, e.g. python/reference-store.md:74, :102, :357; python/reference-streaming-modes.md:16, :248, :589, :684; python/reference-state-graph.md:62, :267, :309, :400; python/langgraph_comprehensive_guide.md:2785–5978 (11 blocks) | Signature listings and pseudo-literals (`<state>`, `Interrupt(...)`) inside ```python fences; not valid Python. | Fence them as `text`/`pyi`, or write real `def ...: ...` stubs. |
| m2 | python/chapter-01-setup-and-core-concepts.md:119 | "`MemorySaver` is deprecated". It is a plain backward-compatible alias (`MemorySaver = InMemorySaver`) with no warning. | "is an older alias; prefer `InMemorySaver`". |
| m3 | python/reference-runtime-and-managed-values.md:184 | Heading "`get_runtime()` — context-manager alternative". It is a plain function. | "function alternative". |
| m4 | python/chapter-05-memory.md:76 | Comment says `setup_tables()`, but the code calls `.setup()`. | Fix the comment. |
| m5 | python/chapter-08-middleware-hooks.md:506; python/chapter-09-advanced-patterns.md:581 | `default_retry_on` has no `httpx.TransportError` branch (it falls through to "retry anything else"). The ch9 table implies it retries only transient errors, when it is a blocklist. | Mirror the source order, and say "everything except the listed programming errors". |
| m6 | python/reference-state-graph.md:400–410 | The `compile()` signature omits `transformers=`. | Add it. |
| m7 | python/langgraph_advanced_error_recovery.md:370, :516 | Two sections numbered "## 6.". | Renumber. |
| m8 | python/langgraph_comprehensive_guide.md:2695, :2702–2723, :2727–2740 | Old docs URL (`langchain-ai.github.io/langgraph/`; docs now at docs.langchain.com), generic "Performance Tips", "Good luck with your AI engineering journey!", and a revision table that stops at 1.2.7 with internal paths (`.routine-envs/...`). | Update the URL, remove the filler and the stale table. |
| m9 | typescript/index.mdx:126–130 | Revision rows have a 4th column "Claude routine" and internal scratch paths (`/tmp/npm-0603`, `.routine-envs/...`). | Trim to three columns and remove the paths. |
| m10 | python/index.mdx:128, :136–:158 area; typescript/index.mdx:35–86 | Relative `./` links. They work on index pages, but AGENTS.md asks for root-absolute links. | Convert to `/langgraph-guide/...`. |
| m11 | python/langgraph_observability_python.md:302–303 | Legacy `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY`. | `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` (as the TS pages do). |
| m12 | python/index.mdx:15; python/langgraph_comprehensive_guide.md:31, :2234 | "Python 3.10 – 3.14": the 1.2.16 wheel classifiers list 3.10–3.13 (`Requires-Python >=3.10`). | "Python 3.10+", or verify 3.14 before claiming it. |
| m13 | python/chapter-01-setup-and-core-concepts.md:29, :647 | Customer-name marketing ("trusted by Klarna, Replit, and Elastic") and a stale requirement (`langchain-core>=0.3.11`). | Drop both. |
| m14 | python/langgraph_advanced_implementations.md:3; typescript/langchain_langgraph_comprehensive_guide.md:3 | Description truncated mid-word ("designed fo"); description is marketing ("Beginner to Expert Level Production-Ready Patterns"). | Rewrite the descriptions. |
| m15 | python/langgraph_performance_optimization.md:2 (title), sidebar label in astro.config.mjs:153; ~14 "optimiz-" and 7 "behavior" uses in Python prose | American spelling against the British English rule. | Optimisation, behaviour. |
| m16 | python/index.mdx:37 vs chapter-01:12 (and similar for other chapters) | Chapter time estimates disagree (15 vs ~20 min, and so on). | Make the times match. |
| m17 | python/index.mdx:196, :213 | Revision text still talks about "per-volume history" and "46 per-class example volumes". | Shorten to "history is in git". |

#### Top 5 improvements

1. **Add executed-code CI for both languages.** Run every non-fragment Python block (with a stub chat model, as in this assessment) and `tsc --strict` + `tsx` every TS block, on each version bump. That would have caught B1–B16 and B22–B23 automatically. Mark fragments and signature listings explicitly so the check can skip them.
2. **Move the agent story to `create_agent` + middleware.** Rewrite ch2 Example 4 and ch8 (Python) around `langchain.agents.create_agent` and the built-in middleware (HITL, summarisation, call limits, PII), and keep `create_react_agent` as a short legacy note. This fixes the biggest currency gap and the "middleware vs hooks" contradictions (B17, M1).
3. **Consolidate to one canonical page per topic.**
   - Delete or auto-generate the 6,000-line comprehensive page.
   - Fold reference-managed-values into reference-runtime-and-managed-values.
   - Keep the policies on one reference page linked from the chapters.
   - Delete langgraph_advanced_implementations, and merge the duplicate recipes.
   - Add a TS sidebar group.
4. **Fix the false behavioural claims as a batch.** Recursion limit 10007; the real `ToolNode` error default; interrupt gotchas; InMemoryStore has no TTL; `destinations=` validates at compile; `durability` is an `invoke` argument and not required for interrupts; the Errata section. Then fix the stale model IDs and "LangGraph Cloud"/`auth login`.
5. **Bring TypeScript up to parity.**
   - Chain all builders so examples pass `strict`, and ship a working `tsconfig`.
   - Drop `@langchain/community`.
   - Add interrupt/resume, `Send`, `Command`, subgraphs, Functional API and testing sections.
   - Replace the invented type-hierarchy diagram.

---

### CrewAI guide assessment (`src/content/docs/crewai-guide/`)

Assessed 2026-10-21 against `crewai==1.15.28` / `crewai-tools==1.15.28` (the versions in `src/data/versions.json`; also the latest on PyPI, released 2026-10-14). Venv: `scratchpad/crewai/venv` (Python 3.13). Scratch work: `scratchpad/assess-crewai/` (`blocks/` = extracted code, `results.tsv` = run results, `runs/*.out` = per-block logs, `check.py` / `req.py` = static checks, `forkrun.py` = mock-LLM runner).

#### Scorecard

| # | Criterion | Score | One-line justification |
|---|---|---|---|
| 1 | Correctness of code | 2/5 | The seven reference pages (Agent/Task/Tools/Memory/Knowledge/LLM/CLI) are mostly right. Every Flow router example fails at class definition, and the comprehensive, flows, recipes and production pages contain many blocks that raise on construction (16 runtime failures, 40+ `Agent` calls without the required `backstory`). |
| 2 | Currency | 2/5 | Version placeholders are clean. But `gpt-4-turbo` appears about 30 times, alongside `gpt-3.5-turbo`, `claude-3-opus`/`claude-3-sonnet`, `llama2`, 2024 pricing and an `api_version="2024-02-15-preview"`. No page uses a Claude 5 or GPT-5.x ID in an example. |
| 3 | Accuracy of prose | 2/5 | False claims: a "Custom Process", "exactly one `@start()`", multi-argument `@listen` as fan-in, a `share_crew_state` field, the context-isolation rule, `CrewOutput.to_dict()`, `@tool` needing annotations, the Qdrant Edge install command, and 1.14.0 not being marked as yanked. Plus hype ("revolutionary 2025 feature", "exceptionally powerful", "10-100x faster"). |
| 4 | Coverage | 3/5 | Good on Agent fields, MCP, A2A, checkpoints, the CLI, Memory and Knowledge. Missing for Flows: structured state (`Flow[Model]`), `@persist`, `and_`/`or_`, `human_feedback`, conversational and declarative flows. Also missing: event listeners and tracing, `kickoff_async`/`kickoff_for_each`, LLM hooks. Padded with UV marketing, backstory essays and repetitive recipes. |
| 5 | Structure and navigation | 2/5 | The index has install, a working example and a page map. But only the index is in the sidebar, so 13 pages are reachable only from index cards. There are 51 relative `./` links (AGENTS.md requires root-absolute), a "deep dive" label, Flows covered twice, a stub MCP section, and an advanced multi-agent page that is a 51-line outline. |
| 6 | Consistency | 2/5 | Two generations of content: the reference pages (accurate, `gpt-4o-mini`) and the legacy pages (`gpt-4-turbo`, invented APIs). The diagrams page shows the removed short-term/long-term/entity memory. Six pages lack `language` in frontmatter. Tone is uneven. |
| 7 | Production readiness | 1/5 | Generic filler: hand-rolled caches and loggers, 2024 price tables, a `Fernet` key "manager", a Dockerfile that bakes `.env` into the image, and invented `crew.execute_task`/`task.execute`. Nothing on CrewAI's own event bus, tracing, `crewai deploy validate`, checkpoint recovery, guardrails or mock-LLM testing. |

**Verdict.** The guide is two guides stuck together. The 1.14–1.15 reference pages are largely accurate and useful. The older comprehensive, flows, recipes, production and diagrams pages (about 6,300 of 8,600 lines) predate the Pydantic-based Flow rewrite and current models. Many of their examples crash on construction, and the Flow router pattern taught everywhere is rejected by 1.15.28. Until those pages are rewritten or cut back, a newcomer following "Zero → Hero" will hit errors in step 1 and step 10.

#### Code verification

Method:
- Extracted all 267 fenced blocks: 206 Python, 61 bash/yaml/text/dockerfile.
- Python: compiled with `compile()`, then statically checked imports and keyword arguments against real signatures and Pydantic fields (`check.py`) and required fields (`req.py`).
- Executed every Python block in a forked child with a mock `BaseLLM` patched over `crewai.LLM` (returns a canned `Final Answer`), stdin closed, 40 s alarm.
- Bash: CLI commands and flags checked against `crewai <cmd> --help` for 1.15.28.

Result classes:
- **OK**: ran to completion.
- **FAIL**: a real error caused by the guide's code.
- **FRAG**: an intentional fragment (undefined names or `...`), checked statically instead.
- **ENV**: needs a real key, file or optional package (`OPENAI_API_KEY` for memory/knowledge embeddings, `pytest`, `fastapi`, `sqlalchemy`, `docling`, `a2a`, `langchain_community`, PyGithub, sample PDFs).
- **MOCK**: mock artefact (structured output from the canned reply); not counted against the guide.

| Page | Py blocks | OK | FAIL (runtime) | FRAG | ENV | MOCK | Extra static failures (in FRAG/OK blocks) |
|---|---|---|---|---|---|---|---|
| index.mdx | 1 | 1 | 0 | 0 | 0 | 0 | 0. The minimal example runs end to end with the mock. |
| crewai_comprehensive_guide.md | 67 | 15 | 5 | 45 | 2 | 0 | 12 `Agent` calls without `backstory`, 13 `Task` calls without `expected_output`, `share_crew_state=`, `json.loads(CrewOutput)`, a hierarchical crew without a manager, a router snippet. About 25 blocks are broken. |
| crewai_flows_guide.md | 23 | 10 | 9 | 3 | 1 | 0 | Router snippet L242. |
| crewai_agent_api_python.md | 13 | 6 | 0 | 5 | 2 | 0 | 0 |
| crewai_task_api_python.md | 17 | 0 | 0 | 16 | 0 | 1 | Guardrail lambda replaces the output (L282). |
| crewai_tools_python.md | 14 | 5 | 0 | 8 | 1 | 0 | Dynamic MCP filter signature wrong (L304). |
| crewai_memory_python.md | 13 | 1 | 0 | 11 | 1 | 0 | Embedder config key in prose (L36). |
| crewai_knowledge_python.md | 15 | 1 | 0 | 8 | 6 | 0 | `NotionSource` is abstract (L127); embedder key wrong (L167). |
| crewai_llm_providers_python.md | 12 | 3 | 0 | 9 | 0 | 0 | 0 |
| crewai_cli_python.md | 2 (+15 bash, 3 yaml) | 1 | 0 | 0 | 1 | 0 | `main.py` missing `import sys` (L150). All CLI commands and flags exist. |
| crewai_recipes.md | 10 | 5 | 1 | 0 | 3 | 1 | Recipe 2 defines `llm` but never uses it and never interpolates `data_path`. |
| crewai_production_guide.md | 18 | 5 | 1 | 7 | 5 | 0 | `share_crew_state=`, 4 agents without `backstory`, `FileReadTool` name, `task.execute`, `crew.execute_task`/`task_count`, `isinstance(result,(str,dict))`. |
| crewai_advanced_multi_agent_python.md | 1 | 0 | 0 | 0 | 1 | 0 | 0. It runs with a real key (no `llm` given, so the default OpenAI model is used). |
| crewai_diagrams.md | 0 (17 ASCII) | – | – | – | – | – | – |
| **Total** | **206** | **53** | **16** | **112** | **23** | **2** | **About 45 more blocks broken by static checks** |

Pass/fail summary: 206 Python blocks checked. 53 ran clean, 16 failed at runtime, and about 45 more fail static checks (missing required fields, non-existent fields or methods, wrong signatures). That is roughly 61 broken blocks, about 30% of all Python blocks; 52 of them are on the four legacy pages. The rest are fragments or need a real API key or optional dependency. Bash and CLI blocks: all commands and flags exist in 1.15.28.

Runtime failures (all reproduced; logs in `scratchpad/assess-crewai/runs/<block>.out`):

| Block (file:line of fence) | Error |
|---|---|
| comprehensive L1833, L2091; flows L434, L487, L911, L1071 | `ValidationError: Invalid flow definition … listen condition 'handle_urgent' references the handler name 'handle_urgent'. A listener triggered by its own completion …` (router label equals the listener's method name) |
| comprehensive L1937; flows L694 | `AttributeError: property 'state' of 'StatefulFlow' object has no setter` (`self.state = {...}` in `__init__`) |
| flows L986, L1258 | `TypeError: listen() takes 1 positional argument but 3 were given` (`@listen(a, b, c)`) |
| flows L160, L1178 | `self.initial_state` is `None` → `AttributeError` / `TypeError` |
| comprehensive L1639, L1998 | `ValidationError: Agent backstory Field required` |
| recipes L258 (Recipe 3) | `ValidationError: manager_llm or manager_agent is required when using hierarchical process` |
| production L593 | `SyntaxError: 'return' outside function` |

#### Issues, ranked

### Blocker

1. **crewai_flows_guide.md:242, 459–478, 516–538, 958–972, 1092–1124, 1308–1330 and crewai_comprehensive_guide.md:1857–1873, 2131–2139.** Every router example names the route label after the listener method (`@listen('handle_urgent') def handle_urgent`). 1.15.28 rejects this when the class is defined. *Fix:* use distinct labels (`return "urgent"` / `@listen("urgent") def handle_urgent`), and say why in one sentence.
2. **crewai_flows_guide.md:700–709; crewai_comprehensive_guide.md:1943–1950.** `self.state = {...}` in `__init__` raises `AttributeError`. *Fix:* teach `class MyFlow(Flow[MyState])` with a Pydantic state model, or unstructured `self.state["k"] = v` without assigning the property.
3. **crewai_flows_guide.md:1051, 1332.** `@listen(a, b, c)` raises `TypeError`, and the text calls it "fan-in". *Fix:* `@listen(and_(a, b, c))` for fan-in, `@listen(or_(a, b, c))` for "any".
4. **crewai_flows_guide.md:170, 206, 1196.** `self.initial_state` is not the kickoff inputs (it is `None` / the state type). *Fix:* read `self.state["name"]` or a typed state field populated by `kickoff(inputs=...)`.
5. **crewai_flows_guide.md:193.** "Every flow must have exactly one `@start()`" is false; several start methods are allowed. *Fix:* "one or more `@start()` methods".
6. **crewai_comprehensive_guide.md:490–653, 1070–1078, 1583–1596, 1643–1646, 2012–2073; crewai_flows_guide.md:1015; crewai_production_guide.md:174, 186, 196, 209.** About 25 `Agent(...)` calls omit `backstory`, which is required (`ValidationError`). *Fix:* add `backstory=` to every example.
7. **crewai_comprehensive_guide.md:994–1060, 1085–1087, 1416–1432.** Thirteen `Task(...)` calls omit the required `expected_output`. *Fix:* add `expected_output=`.
8. **crewai_comprehensive_guide.md:1486; crewai_production_guide.md:51.** `share_crew_state=True` is not a `Crew` field (`ValidationError: extra inputs`); "Optimise for reliability" is invented. *Fix:* delete the argument.
9. **crewai_comprehensive_guide.md:284.** "Custom Process: user-defined execution logic". `Process` has only `sequential` and `hierarchical` (the Task page itself says so at task_api:196–204). *Fix:* delete the item.
10. **crewai_comprehensive_guide.md:911–936 and crewai_recipes.md:311–316.** Hierarchical crews without `manager_llm`/`manager_agent` raise `ValidationError`. *Fix:* add `manager_llm=LLM(model="openai/gpt-5.4")` (or a `manager_agent`).
11. **crewai_comprehensive_guide.md:1307.** `json.loads(result)` on a `CrewOutput` raises `TypeError`. *Fix:* `result.json_dict` (with `output_json=`) or `json.loads(result.raw)`.
12. **crewai_comprehensive_guide.md:1629–1649.** The MCP section is a stub. It shows an `Agent` without `backstory`, claims MCP "requires mcp-python-sdk", says "MCP tools automatically integrated through configuration", and claims agents are exposed as MCP tools. *Fix:* replace it with a link to `/crewai-guide/crewai_tools_python/#mcp-server-tools--agentmcps` plus a three-line `mcps=[MCPServerStdio(...)]` example.
13. **crewai_memory_python.md:36 and crewai_knowledge_python.md:167.** The embedder config `{"model": "text-embedding-3-small"}` raises `TypeError: OpenAIEmbeddingFunction.__init__() got an unexpected keyword argument 'model'`, so the advertised upgrade fix does not work. *Fix:* `{"provider": "openai", "config": {"model_name": "text-embedding-3-small"}}`.
14. **crewai_knowledge_python.md:122–139.** "Two methods to implement". `BaseKnowledgeSource` also has abstract `aadd`, so `NotionSource(...)` raises `TypeError: Can't instantiate abstract class`. *Fix:* add `async def aadd(self) -> None` and say "three methods".
15. **crewai_tools_python.md:304–305.** The dynamic filter is written as `(tool_name: str, metadata: dict)`. The real signature is `(context: ToolFilterContext, tool: dict)`. *Fix:* `def allow_read_only(ctx, tool): return tool.get("annotations", {}).get("readOnlyHint", False)` (and state the signature).
16. **crewai_task_api_python.md:282.** `guardrail=lambda o: (len(o.pydantic.steps) >= 3, "need ≥3 steps")`: on success, the task's raw output is replaced by the string "need ≥3 steps" (task.py sets `task_output.raw = result`). *Fix:* `lambda o: (True, o) if len(o.pydantic.steps) >= 3 else (False, "need ≥3 steps")`.
17. **crewai_task_api_python.md:152.** "If you forget `context=[...]` … `summary` sees nothing" is false. The default context aggregates all prior task outputs, and a sync task waits for pending async futures (crew.py `_get_context`, `_process_async_tasks`). Lines 137 and 213 ("defaults to the previous task") are also inaccurate. *Fix:* "defaults to all previous task outputs; use `context=[...]` to narrow it".
18. **crewai_production_guide.md:593–640.** `return` outside a function (`SyntaxError`). *Fix:* wrap in a function or remove the block; prefer CrewAI's own `cache=` or an LLM hook.
19. **crewai_production_guide.md:654, 921–929.** `task.execute`, `crew.execute_task` and `crew.task_count` do not exist. *Fix:* use `crew.kickoff_for_each(inputs=[...])` / `akickoff_for_each`, `Task.execute_sync`, or delete the "load balancer".
20. **crewai_production_guide.md:341.** `COPY .env .` bakes secrets into the image, on the page meant to cover security. *Fix:* drop the line; pass secrets at runtime (`env_file`, Kubernetes Secrets, as already shown at :445).
21. **crewai_production_guide.md:181, 265.** `agent.tools[0].name == "read_file"` (the real name is "Read a file's content") and `isinstance(result, (str, dict))` (it is a `CrewOutput`) both fail. These tests also call a real LLM. *Fix:* assert `isinstance(result, CrewOutput)`, and use a stub `BaseLLM` like the one in `scratchpad/assess-crewai/forkrun.py`.

### Major

1. **Stale model IDs (examples presented as current).**
   - `gpt-4-turbo`: comprehensive 197, 303, 319, 364, 387, 618, 642, 649, 656, 678, 866, 969, 1078, 1090, 1466, 1763, 2007; flows 275, 337, 631, 995, 1267, 1354; production 70, 729, 788; recipes 46, 209.
   - `gpt-3.5-turbo`: comprehensive 615, 865; production 71, 730.
   - `anthropic/claude-3-opus`: comprehensive 330 (not even a valid API ID).
   - `claude-3-opus`/`claude-3-sonnet` with 2024 prices: production 727–732.
   - `openrouter/anthropic/claude-3-5-sonnet`: comprehensive 1238.
   - `ollama/llama2`: comprehensive 341.
   - `groq/llama3-70b`: llm 262.
   - `claude-sonnet-4-6`: llm 118 (superseded by `claude-sonnet-5-5`).
   - `claude-haiku-4-5`: comprehensive 2216.
   - `gpt-4o` / `gpt-4o-mini` everywhere on the reference pages and index:125 (still listed, but not current; CrewAI's own memory default is `gpt-5.4-mini` and its `DEFAULT_LLM_MODEL` is `gpt-4.1-mini`).
   - `o1` / `o3` / `o4` as "reasoning models": llm 103, 115.
   - Azure `api_version="2024-02-15-preview"`: comprehensive 352.

   *Fix:* use `openai/gpt-5.4-mini` / `openai/gpt-5.4`, `anthropic/claude-sonnet-5-5` / `claude-haiku-5-5` / `claude-opus-5-5`, and `ollama/llama3.1`. Note that `anthropic/...` needs `crewai[anthropic]`; comprehensive:326–335 does not say so, and without it the call raises `ImportError`.
2. **Sidebar** (`astro.config.mjs:146`). CrewAI is a single link, so none of the 13 sub-pages appear in the sidebar even though they carry `sidebar.order`. *Fix:* make CrewAI a collapsed group (Overview, Comprehensive, Agent API … Diagrams), like the LangGraph and AutoGen entries.
3. **Flows coverage** (crewai_flows_guide.md, whole page). None of the current Flow API appears: `Flow[StateModel]`, `self.state` from `kickoff(inputs=)`, `@persist`, `and_`/`or_`, `human_feedback`, `flow_config`, `plot()`/`crewai flow plot`, conversational flows (`ConversationalConfig`, stable since 1.15.18), declarative definitions. *Fix:* rewrite around structured state plus these features, and cut the six near-identical crew-in-flow examples.
4. **Duplicated Flows content.** comprehensive:1653–2192 (540 lines) repeats the flows page, including the same broken patterns. *Fix:* reduce it to a short overview and link `/crewai-guide/crewai_flows_guide/`.
5. **crewai_diagrams.md:348–390, 46–48.** The memory diagram shows short-term/long-term/entity memory, which the memory page (memory:13, 241) says were folded into unified `Memory`. No Flow diagram exists. *Fix:* redraw as Memory → scope/slice → LanceDB/Qdrant Edge, and add a Flow start/listen/router diagram.
6. **crewai_production_guide.md, whole page.** None of CrewAI's own tooling: event listeners (`crewai.events`), `crewai traces enable`, OpenTelemetry, `CrewOutput.token_usage` / `usage_metrics` for cost, `max_rpm`/`max_execution_time`, checkpoints for recovery, `tool_failure_policy`, guardrails, `crewai deploy validate`, `crewai test`/`crewai eval` in CI. *Fix:* rebuild the page around these; delete `LLMCache`, `OptimisedMemoryManager`, `SecureKeyManager`, `CostTracker`, `CrewLoadBalancer`.
7. **crewai_advanced_multi_agent_python.md.** 51 lines, one example, two-bullet "Observability". The index lists it as "30 min". *Fix:* expand it (`allow_delegation`, `manager_agent`, `planning`, A2A delegation, HITL with `human_input`/`human_feedback`) or merge it into the Task API page.
8. **Relative links.** 51 `./` links (index.mdx ×43, comprehensive ×5, flows ×2, knowledge ×1) contradict AGENTS.md ("use root-absolute internal links"). `check-links` passes only because they resolve. *Fix:* `/crewai-guide/<page>/`.
9. **crewai_comprehensive_guide.md:1522–1524; index.mdx:117.** "1.14.0 introduces …" and only "v1.10.0 yanked". PyPI shows **1.14.0 is also yanked** (released 2026-04-07). *Fix:* add "1.14.0 yanked — use 1.14.1+".
10. **crewai_tools_python.md:84, 334.** "Function must have typed parameters … raises ValueError if either is missing". Only a missing docstring raises; unannotated parameters are accepted. *Fix:* reword ("docstring required; annotations strongly recommended for the schema").
11. **crewai_memory_python.md:64.** Qdrant Edge install is given as `pip install qdrant-client`. The backend uses `qdrant-edge-py` via `crewai[qdrant-edge]`. *Fix:* `pip install "crewai[qdrant-edge]"`.
12. **crewai_task_api_python.md:244.** "`.to_dict()` — dict of all outputs keyed by task name". It returns only the final task's `json_dict`/`pydantic` dump. *Fix:* correct the description.
13. **crewai_comprehensive_guide.md:1631–1635, 1657; crewai_flows_guide.md:36, 54, 1460.** "Revolutionary 2025 feature", "Flows solve all these limitations", "represents the future". These also contradict flows:110 ("experimental … autumn 2024"). *Fix:* plain description with the real history.
14. **crewai_cli_python.md:310 vs 319.** `crewai chat` "Needs `chat_llm`" vs "falls back to the crew's default LLM". *Fix:* pick the true behaviour (check `crewai_cli` chat).
15. **crewai_comprehensive_guide.md:1151–1170.** `GithubSearchTool` fails without PyGithub (`No module named 'github'`), which `crewai[tools]` does not install. *Fix:* note `pip install PyGithub`, or use a different example.

### Minor

1. **Frontmatter `language: python` missing**: comprehensive_guide, flows_guide, recipes, production_guide, diagrams, advanced_multi_agent. *Fix:* add it.
2. **index.mdx:38.** "Agent API deep dive" label (banned framing); index.mdx:30 "The new API reference pages". *Fix:* "Agent API reference"; drop "new".
3. **Hype and unverifiable numbers**: comprehensive:38 ("exceptionally powerful … seamless"), :94, :99, :2245, :2274 ("10-100x faster", "blazing fast"); flows:41, 62, 1466 ("seamlessly"). *Fix:* remove.
4. **UV padding**: comprehensive:82–102 and 2239–2356 (about 140 lines of generic `uv pip` usage, duplicated). *Fix:* keep one short `crewai install` / `uv sync` note.
5. **Backstory essays**: comprehensive:426–565 (seven long persona examples). *Fix:* keep one.
6. **Fragments with `...` that do not parse**: agent_api:234, task_api:143, tools:289, knowledge:226. *Fix:* use `# role=..., goal=...` comments or full kwargs.
7. **crewai_cli_python.md:150–163.** `main.py` uses `sys` without importing it, and `inputs={...}` (a set). *Fix:* `import sys`, `inputs={"topic": "AI agents"}`.
8. **crewai_comprehensive_guide.md:846, 1268–1291** and similar. "breakthroughs from 2024" and other stale dates in examples. *Fix:* use undated wording.
9. **Release dates**: comprehensive:2374 says 1.14.1 is April 9, 2026; PyPI says 2026-04-08. Mixed date formats across the revision tables (index uses ISO, comprehensive uses "October 21, 2026"). *Fix:* ISO dates throughout.
10. **crewai_agent_api_python.md:91 vs 297.** "Setting [deprecated fields] still works but emits a DeprecationWarning" vs "`allow_code_execution` is a no-op". *Fix:* reconcile.
11. **crewai_recipes.md:162–249.** Recipe 2 creates `llm` but never passes it; `data_path` is never referenced in any task; the return annotation says `AnalyticsReport` but the function returns a `CrewOutput`. *Fix:* pass `llm=`, use `{data_path}` in a description, return `result.pydantic`.
12. **crewai_diagrams.md:580–620.** ASCII boxes are misaligned (`├────┤` widths differ). *Fix:* tidy, or convert to SVG/Mermaid.

#### Top 5 improvements

1. **Rewrite the Flows material against 1.15.28** (flows guide plus comprehensive §14): structured `Flow[State]`, distinct router labels, `and_`/`or_`, `@persist`, `human_feedback`, conversational and declarative flows. Run every example with a stub `BaseLLM` before publishing. This removes 9 runtime blockers and the largest coverage gap.
2. **Fix or cut the legacy comprehensive guide.** Add the required `backstory`/`expected_output`, delete `share_crew_state`, "Custom Process" and the MCP stub, and replace the UV and backstory padding with links to the accurate reference pages. That would take the page from about 2,400 to about 900 lines.
3. **Replace the production guide** with CrewAI-specific practice: event listeners and tracing, `token_usage`-based cost reporting, checkpoints and `tool_failure_policy` for recovery, guardrails, secrets handled outside the image, `crewai deploy validate` + `crewai test`/`eval` in CI, and mock-LLM unit tests.
4. **Do one model-ID sweep across all pages** (`gpt-5.4-mini`/`gpt-5.4`, `claude-sonnet-5-5`/`claude-haiku-5-5`/`claude-opus-5-5`, current Gemini and Ollama names), with a note that Anthropic, Gemini and Bedrock need `crewai[anthropic]` etc. Delete the 2024 price table.
5. **Fix navigation and house rules:** a sidebar group for all 14 pages, root-absolute links, `language` frontmatter, no "deep dive" label. Update the diagrams to unified Memory and add a Flow diagram.

---

### Google ADK guide: assessment (2026-10-21)

Scope: `src/content/docs/google-adk-guide/` (top-level `index.mdx`, `python/` 16 pages, `go/` 4 pages, `typescript/` 1 page). Versions from `src/data/versions.json`: `google-adk` 2.12.0, `google.golang.org/adk` 1.9.0, `@google/adk` 2.3.0. I confirmed all three are the latest on PyPI, the Go proxy and npm today.

Scratch work: `/tmp/claude-0/-home-user-AgentGuides/55b0d51f-358a-5617-8094-250bd4df3d90/scratchpad/assess-google-adk/` (mock-LLM harness `run/mock.py`, block runner `run/runblock.sh`, results `py.out`, `run_results.txt`, Go module `go/`, TS project `ts/`).

#### Scorecard

| # | Criterion | Score | Justification |
|---|---|---|---|
| 1 | Correctness of code | 3/5 | Go is clean: 40/40 programs vet and the fake-model test passes. TS type-checks and runs. Python imports and kwargs are almost all right, but 13 Python blocks fail at runtime or are semantically wrong: 6 skip session creation, plus an invented Agent Engine API and the wrong Claude class. |
| 2 | Currency | 2/5 | The placeholders are clean. But `gemini-2.5-*` IDs appear 327 times, and the Gemini API now limits 2.5 to existing users. Claude IDs are 4.x, OpenAI is `gpt-4o`, `OpenAILlm` is shown at its old `labs` path, and the pre-2.0 workflow agents are taught first. |
| 3 | Accuracy of prose | 3/5 | Mostly checked against source, with good detail. Some claims are false: the "direct Anthropic" section uses the Vertex class, "2.5-flash is widely available", a `session:` state prefix that does not exist, and wrong release dates. The production guide has invented cost and quality scores. |
| 4 | Coverage | 2/5 | Python has wide API coverage but almost nothing on the `adk` CLI (create/run/web/eval/deploy) or `AdkApp`/Agent Engine deployment. TypeScript is a single page. The comprehensive guide carries about 3,850 lines of class-internals reference. |
| 5 | Structure & navigation | 2/5 | The sidebar lists only 4 ADK links, so 18 sub-pages are reachable only from index cards. 18 relative links on non-index pages will resolve under the wrong directory. Three overlapping changelogs. "Class & API Reference / verified 2.7.1 / Why it matters / Correction" framing remains. |
| 6 | Consistency | 3/5 | Frontmatter is complete and no page opens with an H1. The plugin hook count is given as 12, 13 and 14 on different pages. Release dates differ between tables. Python prose has 29 US spellings and emoji in the card titles. |
| 7 | Production readiness | 2/5 | Go production is current and sensible (Artifact Registry, `--no-allow-unauthenticated`, `telemetry.New`). The Python production guide has an invented Agent Engine deployment, plus `gcr.io`, `--allow-unauthenticated` and generic retry/circuit-breaker filler. It never mentions `adk deploy`. |

**Verdict.** The Go guide is in good shape and the TypeScript page is correct but thin. The Python guide is large and mostly import-accurate, but a new developer hits real problems:
- The default model in nearly every example is restricted for new projects.
- Several "first agent" runner snippets raise `SessionNotFoundError`.
- The Agent Engine deployment and direct-Claude sections are wrong.
- The comprehensive guide is still dominated by a folded-in class reference.

Fixing the model IDs, the session bug class and the production deployment section would lift the guide the most.

#### Code verification

How I tested:
- **Python:** google-adk 2.12.0 in a venv. An AST checker covered every `python` block: does it parse, do `google.adk` imports resolve, and do constructor and method keyword arguments match the installed signatures. Every block containing `asyncio.run(` was then executed against a registered mock `BaseLlm` that takes over the `gemini-.*` names.
- **Go:** module pinned to `google.golang.org/adk v1.9.0`, Go 1.26 toolchain, `go vet` per program, and `go test` for the fake-model test.
- **TypeScript:** `@google/adk` 2.3.0 with `tsc --strict` (nodenext). I ran it end to end with a `BaseLlm` subclass mock that makes a tool call.

### Python

| Page | Blocks | Static pass | Static issues | Executed | Run pass | Run fail (real) | Run fail (needs creds/placeholder) |
|---|---|---|---|---|---|---|---|
| index.mdx | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| agents.md | 29 | 29 | 0 | 8 | 8 | 0 | 0 |
| workflows.md | 24 | 24 | 0 | 4 + 1 | 4 | 1 (L655) | 0 |
| tools.md | 49 | 49 | 0 | 1 | 1 | 0 | 0 |
| callbacks-and-plugins.md | 23 | 23 | 0 (BigQuery import needs documented extra) | 0 | – | – | – |
| runner-and-sessions.md | 26 | 26 | 0 | 3 | 2 | 1 (L486) | 0 |
| memory-and-artifacts.md | 12 | 12 | 0 | 4 | 2 | 0 | 2 (GCP ADC) |
| mcp-and-a2a.md | 17 | 16 | 1 signature stub (L122) | 6 | 4 | 0 | 2 (npm registry, key file) |
| evaluation.md | 27 | 27 | 0 | 3 | 0 | 0 | 3 (`my_package`, undefined `eval_set` fragment) |
| google_adk_advanced_python.md | 24 | 24 | 0 (1 semantic: L813–835) | 0 | – | – | – |
| google_adk_iam_examples.md | 10 | 10 | 0 | 0 | – | – | – |
| google_adk_comprehensive_guide.md | 360 | 356 | 4 (L461 `.env` in a python fence; L6463, L7073, L7637 signature stubs) | 3 | 1 | 2 (L300, L780) + 4 more found statically (L833, L1146, L3307, L3510) | 0 |
| google_adk_recipes.md | 24 | 24 | 0 | 13 | 11 | 2 (L1298, L1777) | 0 |
| google_adk_production_guide.md | 22 | 22 | 0 (1 semantic: L163–212 invented API) | 1 | 1 | 0 | 0 |
| google_adk_diagrams.md | 0 (ASCII only) | – | – | – | – | – | – |
| **Total** | **648** | **643** | **5** | **47** | **34** | **6 run + 4 static + 3 semantic = 13** | **7** |

Checker hits that turned out to be fine:
- `FunctionNode(func=…)`, `LiteLlm(api_base=…)` and `Event(state=…, route=…)` are all accepted by the runtime.
- The BigQuery plugin, `AntigravityAgent`, `AgentRegistry` and `GcpAuthProvider` imports need extras, and the text names those extras.

Method-call kwargs (`run_async`, `run_debug`, `create_session`, `get_session`, `search_memory`, `save_artifact`, `load_artifact`) were all correct.

### Go

| Page | Blocks | Full programs | `go vet` pass | Fragments |
|---|---|---|---|---|
| index.mdx | 1 | 1 | 1 | 0 |
| google_adk_go_comprehensive_guide.md | 28 | 22 | 22* | 6 (interface listings, which match `agent.Agent`, `tool.Tool`, `model.LLM` and `session.Session` in v1.9.0) |
| google_adk_go_production_guide.md | 5 | 5 | 5 | 0 |
| google_adk_go_recipes.md | 12 | 12 | 12 | 0 |

\* L1408 is the `_test.go` half of a two-file package (L1389 + L1408). Run together, `go test` passes with the fake `model.LLM`.

### TypeScript

`typescript/index.mdx`: 1 block. It passes `tsc --strict` and ran end to end with a mock `BaseLlm`: the tool was called and the final response printed. Every 2.x claim checked against the package holds: `Workflow`, `FunctionNode`, `JoinNode`, `Edge`, `START`, `requireAgent`, `DebugLoggingPlugin` and `ToolCallIntegrityPlugin` are exported, `autoCreateSession` exists, `LLMAgentWrapper` is gone, and 2.0.0 was published on 2026-08-21. The "Node.js 20.19+" requirement is not in the package's `engines` field, so I could not verify it.

### Checks

- `node scripts/check-versions.mjs | grep google-adk`: clean.
- `node scripts/check-links.mjs`: 0 broken. It resolves `./x/` against the file's directory, though, which hides the browser-side problem in M1 below.

#### Issues

### Blockers

1. **python/google_adk_comprehensive_guide.md:300–340, 780–825, 833, 1146, 3307, 3510.** These blocks build a `Runner` and call `run_async(session_id=…)` without creating the session. The first two were run and raise `SessionNotFoundError: Session not found: session456` / `session_1`. They are the first runner examples a reader meets ("The Runner Class", "Synchronous Execution"). **Fix:** add `await runner.session_service.create_session(app_name=…, user_id=…, session_id=…)`, or pass `auto_create_session=True` to `Runner`. The sync example should use `runner.run(...)`.
2. **python/google_adk_production_guide.md:163–212.** "Vertex AI Agent Engine Deployment" calls `aiplatform_v1.AgentsClient().create_agent(...)` with a made-up config dict. `AgentsClient` does not exist in `google.cloud.aiplatform_v1`. **Fix:** replace it with `adk deploy agent_engine …`, or `vertexai.agent_engines.create(agent_engines.AdkApp(agent=root_agent), requirements=[…])`. Note the new 2.12 option for a custom service account.
3. **python/google_adk_advanced_python.md:813–835.** The section is headed "Anthropic Claude via `AnthropicLlm` (direct)" and sets `ANTHROPIC_API_KEY`, but uses `Claude(...)`. In 2.12 `Claude` is the **Vertex AI** subclass: it requires `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION` and raises otherwise. A bare `"claude-…"` string also resolves to `Claude` (Vertex), not the direct API. **Fix:** use `from google.adk.models.anthropic_llm import AnthropicLlm; AnthropicLlm(model="claude-sonnet-5-5")` for the Anthropic API, and document `Claude` as the Vertex route.
4. **python/google_adk_recipes.md:1298.** `LlmAgent(tools=[pipeline])`, where `pipeline` is a `Workflow`, fails at construction: `Node 'resilient_data_pipeline' does not have an input_schema defined. NodeTool requires an explicit Pydantic input_schema`. **Fix:** give the `Workflow` an `input_schema=` Pydantic model, or run the workflow as the root (`App(root_agent=pipeline)`).
5. **python/google_adk_recipes.md:1777, 1781, 1787, 1791, 1800, 1804; python/google_adk_comprehensive_guide.md:6414.** `types.Part.from_text("…")` is called positionally, which raises `TypeError: Part.from_text() takes 1 positional argument but 2 were given`. **Fix:** `types.Part.from_text(text="…")` or `types.Part(text="…")`.
6. **python/workflows.md:655.** The fan-out/join example ends with `print(events[-1].content.parts[0].text)`. The last event comes from the `summarize` function node: it has `output` and `content=None`, so this raises `AttributeError`. **Fix:** `print(events[-1].output)`, or have `summarize` return `types.Content`.
7. **python/runner-and-sessions.md:486.** `s.update_time` raises `AttributeError`. **Fix:** `s.last_update_time`.
8. **Default model across the guide** (`gemini-2.5-*` 327× across Python/Go/TS, mostly `gemini-2.5-flash` and `gemini-2.5-pro`; e.g. python/index.mdx:96, typescript/index.mdx:51, go/index.mdx:97). The Gemini API models page says 2.5 access is "limited to users who have used the 2.5 models in the past", and recommends 3.5 Flash-Lite or 3.8 Flash for new projects. A new developer's first run of the quick-start can fail. python/google_adk_comprehensive_guide.md:349 says the opposite ("uses `gemini-2.5-flash` … because it is widely available"). **Fix:** switch the examples to `gemini-3.5-flash`, ADK's own `LlmAgent.DEFAULT_MODEL` (with `gemini-3.8-flash` as the `adk create` alternative), and rewrite the model table at L351–356.

### Major

1. **Relative links on non-index pages:** python/agents.md:12, 53 (×2), 100, 768, 1020, 1024, 1044, 1123; tools.md:1412, 1444, 1473, 1717; runner-and-sessions.md:963; memory-and-artifacts.md:224; google_adk_comprehensive_guide.md:2207, 3861. Links such as `](./workflows/)` on `/python/agents/` resolve to `/python/agents/workflows/` in the browser. `remark-base-path` does not rewrite relative links, and `check-links.mjs` resolves them against the file directory, so it misses them. This also breaks AGENTS.md ("use root-absolute links"). I inferred this from the routing and did not confirm it in a build. **Fix:** convert them to `/google-adk-guide/python/<page>/…`. Do the same for the 69 relative links on python/index.mdx and go/index.mdx for consistency.
2. **astro.config.mjs:242–250.** The Google ADK sidebar lists only Overview/Python/Go/TypeScript. All 15 Python sub-pages and 3 Go sub-pages are missing (MAF, for comparison, lists its sub-pages). **Fix:** add nested Python and Go groups.
3. **Class-reference framing left in the comprehensive guide** (python/google_adk_comprehensive_guide.md:3883–7725, about 3,850 lines and 350 `####`/`**Source:**` entries):
   - "(verified 2.7.1)" appears 22×.
   - "Why it matters:" 56×, "**Correction**" 12×.
   - "the bare verification venv" at L7310.
   - Private-internals detail such as `_find_cache_info_from_events` (L7064) and the `NodeStatus` hex colours (L7557).

   python/index.mdx:27 and :52 still sell it as "Step 15 · Class & API reference … first built for 2.7.1". **Fix:** move user-facing material into the topic pages (agents, tools, workflows, runner-and-sessions and so on), delete the internals, and drop the Step 15 card and the aside.
4. **The comprehensive guide teaches deprecated orchestration first** (L1038–1140: `SequentialAgent`, `ParallelAgent`, `LoopAgent` as the main "Agent Orchestration Patterns", with only a comment at L1080). recipes.md also has 5 recipes on deprecated agents (L132–1025). In 2.12 these emit `DeprecationWarning`. **Fix:** lead with `Workflow` (as workflows.md does) and keep one short migration note. Keep the caveat that a `Workflow` cannot yet be an `LlmAgent` sub-agent.
5. **Stale non-Gemini model IDs:**
   - Claude 4.x: advanced_python.md:788, 826 (`claude-opus-4-5`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`), 834, 854; recipes.md:1627; comprehensive_guide.md:7314 (`claude-opus-4-5-20250514`, which never existed).
   - `gpt-4o` / `gpt-4o-mini`: agents.md:153; advanced_python.md:779, 799, 861; recipes.md:1636, 1644; comprehensive_guide.md:7305, 7346, 7362.

   **Fix:** use `claude-opus-5-5` / `claude-sonnet-5-5` / `claude-haiku-5-5` and current OpenAI names.
6. **`OpenAILlm` path is stale** (python/google_adk_comprehensive_guide.md:7339–7363: "labs … install `openai` separately; import path is `google.adk.labs.openai`"). Since 2.10.0 the models live in `google.adk.integrations.openai`, with a `google-adk[openai]` extra and `OpenAIResponsesLlm` / `AzureOpenAIResponsesLlm`. `labs` only re-exports them. **Fix:** update the import, the extra and the class list.
7. **Hardly any ADK CLI / Dev UI coverage.** `adk create`, `adk run`, `adk eval` and `adk deploy` appear nowhere. `adk web` / `adk api_server` appear once, at comprehensive_guide.md:7138, as internals. This is the official quick-start path (agent folder with `root_agent`). **Fix:** add a "Project layout and the adk CLI" section to the Python index or comprehensive guide, and use `adk eval` in evaluation.md.
8. **The Python production guide is generic and partly unsafe:**
   - L55–100 uses `gcr.io` (replaced by Artifact Registry) and `--allow-unauthenticated`, and sets `GOOGLE_PROJECT_ID`, which ADK does not read (it reads `GOOGLE_CLOUD_PROJECT`).
   - L1156–1200 has invented per-model "cost/quality/speed" scores.
   - There is no `adk deploy cloud_run|gke|agent_engine`, no `EventsCompactionConfig` or context caching in the cost section, and no Model Armor / safety plugin.

   **Fix:** model it on the Go production page, which already does all of this correctly.
9. **Plugin hook count disagrees.** callbacks-and-plugins.md:357–366 lists 12 hooks and leaves out `on_agent_error_callback` and `on_run_error_callback`. python/index.mdx:74 says "All 13 plugin hooks"; index.mdx:121 says "11+ hooks". `BasePlugin` in 2.12 has 14, which comprehensive_guide.md:5707 gets right. **Fix:** add the two rows and say 14 everywhere.
10. **runner-and-sessions.md:463** uses `"session:last_city"` as if `session:` were a scope prefix. `State` defines only `app:`, `user:` and `temp:`. **Fix:** use an unprefixed key and say unprefixed keys are session-scoped.
11. **Three overlapping changelogs:** index.mdx:62–73, python/index.mdx:115–131 and comprehensive_guide.md:7726–7742. The last stops at 2.7.1, and they mix documentation edits ("Documented …", "source-verified 10 classes", "Claude routine", `.routine-envs` at index.mdx:69 and comprehensive_guide.md:7736–7737) with package releases. **Fix:** keep one version-level table of package changes in python/index.mdx, keep site edits in the top-level revision history, and delete the comprehensive guide's table.
12. **TypeScript coverage is a single page** (typescript/index.mdx). There are no tools, sessions, workflow, MCP/A2A or deployment pages, only a pointer to the Python guide. **Fix:** add at least a comprehensive page covering `FunctionTool` with Zod, `Runner`/sessions, `Workflow`, MCP and `adk-devtools`.

### Minor

1. Wrong release dates. python/index.mdx:121 and comprehensive_guide.md:7734 give 2.0.0 GA as "May 20, 2026"; PyPI upload was 2026-05-19. python/index.mdx:128 gives 2.7.1 as "Aug 21"; the changelog says 2026-08-17. python/index.mdx:129 gives 2.8.0 as "Aug 31"; it was 2026-08-25. index.mdx:72 gives "April 2026 Go 1.0 GA"; v1.0.0 was tagged 2026-03-20, and go/index.mdx:33 correctly says March. **Fix:** correct the dates.
2. python/index.mdx:125 puts `GEPARootAgentOptimizer` under 2.5.0, but it was created in 2.3.0. **Fix:** move it.
3. Comprehensive guide L461: a `.env` file and Python code share one `python` fence (SyntaxError). **Fix:** split into a `bash`/`ini` block and a `python` block.
4. Signature stubs in `python` fences that do not parse: mcp-and-a2a.md:122; comprehensive_guide.md:6463, 7073, 7637. **Fix:** add `...` bodies or mark them `text`. mcp-and-a2a.md:122 also leaves out `rpc_path`, `task_store`, `agent_executor_factory`, `security_schemes` and `default_skill_security`.
5. google_adk_diagrams.md has bare ```` ``` ```` fences, uneven box art (L12–20) and a "Sequential Agent Workflow" diagram (L75), but no `Workflow` graph or routing diagram. **Fix:** add a Workflow/edges/JoinNode diagram and retitle or remove the deprecated one.
6. Go recipes have pointer-only sections: go/google_adk_go_recipes.md:458–460, 745–749, 753–761. **Fix:** inline short programs or drop the headings.
7. US spellings in prose: comprehensive_guide.md has 25 (e.g. L3892 "favor", L3945 "behavior", L4264 "summarization"), python/index.mdx:121, 128, 129 have 4, and go/google_adk_go_production_guide.md:20 has 1. **Fix:** convert to British spelling.
8. Emoji in the card titles at index.mdx:20, 25, 30 (🐍 🐹 📘), against the understated house style. **Fix:** remove them.
9. Comprehensive guide headings at L3835 ("Graph-Based Agent Workflows") and L3865 ("Session Rewind") are H2s under "Advanced Topics" and missing from the TOC at L13–29. **Fix:** demote them to H3.
10. Top-level index.mdx has no "install + minimal example". It is a language picker, which is acceptable, but the language pages carry the quick-starts. **Fix:** add a one-line `pip install` / `go get` / `npm install` row.
11. The TypeScript claim of "Node.js 20.19+" (typescript/index.mdx:24) is not backed by `engines` in the package. **Fix:** cite the source or drop the number.

#### Top 5 improvements

1. **Replace the default model** with `gemini-3.5-flash` everywhere (Python, Go and TS). Rewrite the model table. Move Claude to 5.x and OpenAI to current names. This fixes the most visible currency and runnability problem in one sweep.
2. **Fix the runnable-code blockers.** Add session creation or `auto_create_session=True` to the 6 runner snippets, `Part.from_text(text=…)`, `last_update_time`, workflow `.output`, the `Workflow`-as-tool `input_schema`, and use `AnthropicLlm` for the direct API. Then add these pages to a mock-model CI run like the one used here.
3. **Rewrite the Python production deployment** around `adk deploy` (Cloud Run / GKE / Agent Engine) and `AdkApp`. Use Artifact Registry and authenticated Cloud Run, and add compaction, context caching and safety plugins to the cost and security sections. Use the Go production page as the template.
4. **Dissolve the "Class & API Reference" block.** Move the user-facing facts into the topic pages, delete the internals and the "verified 2.7.1 / Why it matters / Correction" scaffolding, and lead orchestration with `Workflow`.
5. **Fix navigation:**
   - Add the Python and Go sub-pages to the sidebar.
   - Convert relative links to root-absolute ones.
   - Collapse the three changelogs into one.
   - Add a short "adk CLI and project layout" section.
   - Grow the TypeScript guide beyond one page.

---

### Assessment: PydanticAI guide (`src/content/docs/pydanticai-guide/`)

Assessed 2026-10-21 against **pydantic-ai 2.59.0** (the current PyPI release, released 2026-10-21; matches `src/data/versions.json`), installed with the temporal, dbos, prefect, ag-ui, ui, duckduckgo, web-fetch and logfire extras. 23 pages, 19,356 lines, 719 Python blocks. Read in reading order: index, comprehensive guide, topic pages, recipes, production, diagrams.

#### Scorecard

| # | Criterion | Score | Justification |
|---|---|---|---|
| 1 | Correctness of code | 3/5 | 417 of 719 blocks run cleanly against TestModel, and every quick-start runs. But 13 blocks have real bugs: an invented `run_sync(instructions_prepend=)`, `ToolReturn(content=...)` in three places, `stream_text()` on structured output, `stream_text_sync`, a NameError on `Any`, and `asyncio.run(asyncio.gather(...))`. |
| 2 | Currency | 2/5 | `check-versions` is clean and the V2 API migration is thorough. However about 490 lines use superseded model IDs (`gpt-4o` alone appears 226 times). No example uses `claude-*-5-5`, `gpt-6*` or `gemini-3.x`. The deprecated `TemporalAgent`/`PrefectAgent` wrappers still get worked examples. |
| 3 | Accuracy of prose | 3/5 | The rewritten pages (durable execution, graph, evals, integrations) are accurate. The reference has errors: it says `end_strategy` defaults to `'early'` (it is `'graceful'`), it lists the removed `MCPServer*` classes as current toolsets, the provider-support tables contradict each other, `InstrumentationSettings` gets an invented `event_mode`, and all 16 `file.py:NNN` source citations are wrong. |
| 4 | Coverage | 3/5 | Very broad: nearly every V2 feature has a section. The comprehensive guide is about 4,700 lines of class reference, much of it private internals (`_LazyAsyncClient`, `_RequestParams`, `MessagesBuilder` from `ui._messages_builder`, `SystemPromptRunner`). Observability has no proper home. |
| 5 | Structure & navigation | 2/5 | The sidebar has one link for 23 pages. `pydantic_ai_direct_api.md` is an orphan. One 6,476-line page holds most of the guide. The toolsets page covers Prefixed, Filtered and DeferredLoading toolsets three times each. The production page ends with a "(Continues with ...)" stub. Links check clean. |
| 6 | Consistency | 3/5 | 9 pages lack `language:` frontmatter. The name is spelt both "PydanticAI" and "Pydantic AI". Recipes use the raw OpenAI SDK for embeddings even though there is an Embeddings page. The native-tool provider tables disagree. Editorial residue remains ("**Correction:**", "grep found zero trace", "This page was rewritten for V2"). |
| 7 | Production readiness | 2/5 | Mostly generic FastAPI/K8s/Prometheus/Redis filler. Two blocks are broken: Celery `@app.post`, and an nginx config inside a `python` block. It ignores the framework's own production features: `FallbackModel`, `ConcurrencyLimiter`, `RetryConfig`/tenacity transports, `Instrumentation`, `UsageLimits.cost_limit`, durability capabilities, approval toolsets and SSRF-safe fetch. |

**Overall verdict.** The V2 migration is real: no removed V1 API is presented as runnable code, and the pages rewritten in the last pass are accurate and run. The guide is let down by three things. Its model IDs are a generation old throughout. Most of its content sits in a 6,476-line comprehensive page that mixes a tutorial with a source-dive of private internals. Its production page is generic. A dozen code bugs and several false reference claims (end-strategy default, `MCPServer*`, `ToolReturn`) need fixing before it can be called "verified against 2.59.0".

**Blockers: 13** (11 code, 2 prose/page). Majors: 14. Minors: 16.

---

#### Code verification results

### Method

Scripts are in `scratchpad/assess-pydanticai/`.

- **Extraction:** `extract.py` pulled every fenced `python` block and substituted the `%%version=…%%` placeholders.
- **Static checks** (`static.py`):
  - compile with top-level await allowed;
  - resolve every `import` / `from … import` in the 2.59.0 venv;
  - check keyword arguments against `inspect.signature` for imported callables and for methods on variables built from imported classes.
- **Signature listings** (`stubs.py`): the 67 pseudo-signature blocks that do not parse as Python were compared with the real signatures.
- **Runtime** (`pagerun.py`): every block was executed with:
  - `infer_model` patched to validate the model string and then return `TestModel()`;
  - `ALLOW_MODEL_REQUESTS=False`, so explicitly constructed provider models fail with a clear error instead of calling out;
  - fake API keys;
  - a 20 s timeout per block.
  
  Blocks that raised `NameError` were re-run in the page's accumulated namespace (fragments that depend on earlier blocks).
- **Import resolution:** all imports resolve except `langchain_community`, `faiss` and `celery`. These are third-party extras the pages name, so they are not counted as guide errors.

### Categories

- **Pass:** ran with no exception.
- **Env-limited:** needs a real model, a server or an optional dependency, or failed only because TestModel produced schema-valid nonsense. Examples: "TestModel does not support built-in tools", `NativeOutput` unsupported by TestModel, Temporal server not running, `ModelRetry` on TestModel's fake email.
- **Fragment:** a snippet with top-level `await` or names from the surrounding text.
- **Signature stub:** a pseudo-code `Name(arg, *, …)` or `class X(...)` listing tagged `python`.
- **Fail:** a genuine defect in the guide.

| Page | Blocks | Pass | Env-limited | Fragment | Signature stub | Fail |
|---|---|---|---|---|---|---|
| index.mdx | 1 | 1 | 0 | 0 | 0 | 0 |
| pydantic_ai_advanced_error_testing.md | 31 | 26 | 3 | 1 | 0 | 1 |
| pydantic_ai_advanced_patterns.md | 15 | 14 | 0 | 1 | 0 | 0 |
| pydantic_ai_builtin_tools.md | 34 | 22 | 9 | 1 | 2 | 0 |
| pydantic_ai_comprehensive_guide.md | 371 | 169 | 22 | 8 | 163 | 9 |
| pydantic_ai_concurrency.md | 14 | 14 | 0 | 0 | 0 | 0 |
| pydantic_ai_direct_api.md | 18 | 16 | 0 | 2 | 0 | 0 |
| pydantic_ai_durable_execution.md | 4 | 0 | 3 | 1 | 0 | 0 |
| pydantic_ai_embeddings.md | 22 | 5 | 12 | 5 | 0 | 0 |
| pydantic_ai_evals_2025.md | 5 | 5 | 0 | 0 | 0 | 0 |
| pydantic_ai_graph_support.md | 3 | 3 | 0 | 0 | 0 | 0 |
| pydantic_ai_hooks.md | 17 | 14 | 0 | 2 | 0 | 1 |
| pydantic_ai_integrations_2025.md | 7 | 3 | 4 | 0 | 0 | 0 |
| pydantic_ai_message_history.md | 17 | 11 | 0 | 6 | 0 | 0 |
| pydantic_ai_model_providers.md | 30 | 24 | 3 | 1 | 2 | 0 |
| pydantic_ai_output_types.md | 23 | 16 | 6 | 1 | 0 | 0 |
| pydantic_ai_production_guide.md | 8 | 5 | 1 | 0 | 0 | 2 |
| pydantic_ai_recipes.md | 14 | 10 | 4 | 0 | 0 | 0 |
| pydantic_ai_streaming.md | 15 | 7 | 1 | 7 | 0 | 0 |
| pydantic_ai_streaming_server_fastapi.md | 7 | 7 | 0 | 0 | 0 | 0 |
| pydantic_ai_testing.md | 18 | 16 | 0 | 2 | 0 | 0 |
| pydantic_ai_toolsets.md | 45 | 29 | 0 | 16 | 0 | 0 |
| **Total** | **719** | **417** | **68** | **54** | **167** | **13** |

### Quick-start examples

Index minimal example, comprehensive "Hello World" (l.150), output-types minimal (l.14), toolsets minimal, testing minimal, graph (all 3), evals (all 5): all run end to end with TestModel.

Caveat: "pass" means the block executed. Tool bodies only run when TestModel calls them, so a broken tool body can still pass (see `ctx.all_messages()` below).

### Signature-listing mismatches (from `stubs.py`)

- `InstrumentationSettings(event_mode=...)` is not a parameter (comprehensive l.5030).
- `*_model_profile(name)`: the real parameter is `model_name` (l.3106).
- `TextOutput(output: …)`: the real parameter is `output_function` (l.4249).
- `ToolOutput` argument table says `output`; the real parameter is `type_` (output_types l.75).
- `Agent(...)` stub says `end_strategy='early'` and `deps_type=NoneType`; the real defaults are `'graceful'` and `object` (l.1755, l.1764).
- `ToolCallEvent(..., event_kind)` and `ToolResultEvent(..., event_kind)` are not fields (l.4104).

---

#### Issues (ranked)

### Blocker

1. **comprehensive_guide.md:739–742.** `agent.run_sync(..., instructions_prepend='...')` raises `TypeError: unexpected keyword argument 'instructions_prepend'`. **Fix:** use `instructions='Explain at a beginner level…'`.
2. **comprehensive_guide.md:575 (block l.561).** `-> dict[str, Any]` is evaluated before `Any` is imported (the import sits inside the function at l.593), so decoration raises `NameError`. **Fix:** move `from typing import Any` to the top of the block.
3. **comprehensive_guide.md:784–786.** `stream_text()` on an agent with `output_type=Article` raises `UserError: stream_text() can only be used with text responses`. **Fix:** use `async for partial in response.stream_output(): ...`.
4. **comprehensive_guide.md:3477, :6286; hooks.md:347.** `ToolReturn(content=...)` raises `TypeError: missing 'return_value'`. **Fix:** `ToolReturn(return_value='42')`, or pass the plain value in `DeferredToolResults(calls={id: '42'})`.
5. **comprehensive_guide.md:1964.** `result.all_messages(output_tool_return_content=...)` on a `str` output raises `ValueError: Cannot set output tool return content when the return type is str`. **Fix:** give the example a structured `output_type`, or drop the argument.
6. **comprehensive_guide.md:4032–4034.** `result.stream_text_sync(delta=True)` does not exist. **Fix:** `for chunk in result.stream_text(delta=True):` (on `StreamedRunResultSync` this is already sync).
7. **comprehensive_guide.md:4783.** `asyncio.run(asyncio.gather(...))` raises `ValueError: a coroutine was expected`. **Fix:** wrap it in `async def main(): await asyncio.gather(...)`, then `asyncio.run(main())`.
8. **comprehensive_guide.md:3567.** `ModelSettings(tool_choice='required')` is shown as a working call, but 2.59.0 raises `UserError` because output tools would be excluded (model_providers.md:150 says so). **Fix:** delete that line, or mark it as raising.
9. **advanced_error_testing.md:559–560.** `model.agent_model_requests` / `agent_model_responses` do not exist on `TestModel` (`AttributeError`). **Fix:** use `model.last_model_request_parameters` together with `capture_run_messages()` / `result.all_messages()`.
10. **comprehensive_guide.md:252.** `ctx.all_messages()` does not exist on `RunContext`, so the tool raises `AttributeError` as soon as the model calls it. **Fix:** delete it; `ctx.messages` already holds the history.
11. **production_guide.md:787–826.** `app = Celery(...)` is then used as `@app.post(...)` / `@app.get(...)`. Celery has no route decorators and no FastAPI app exists in the block. `store_result` is also undefined. **Fix:** create `api = FastAPI()` for the routes and define or import `store_result`.
12. **production_guide.md:845.** The page ends with "(Continues with more production patterns, best practices, and operational guidance)", a placeholder stub. **Fix:** remove it and write the missing sections (see Major 9).
13. **toolsets.md:51.** The catalogue lists "`MCPServer*` (in `pydantic_ai.mcp`) — toolsets backed by MCP stdio/SSE/HTTP". These classes were removed in 2.0: `pydantic_ai.mcp` exports only `MCPToolset`/`MCPToolsetClient`/FastMCP. **Fix:** replace the row with `MCPToolset` (`pydantic_ai.mcp`).

### Major

1. **Stale model IDs, about 490 lines across 22 files.** No example uses the current Claude 5 family (`claude-opus-5-5` / `claude-sonnet-5-5` / `claude-haiku-5-5`, all in 2.59.0's `KnownModelName`). OpenAI examples use `gpt-4o` (226×), `gpt-4o-mini` (35×), `gpt-5.2` (61×), `gpt-4.1*`, `o3-mini`; the current line in `KnownModelName` is `gpt-6-sol` / `gpt-6.1-sol` / `gpt-5.6-*`. Gemini examples use `gemini-2.5-*` / `gemini-2.0-flash` instead of `gemini-3.5-flash` / `gemini-3.1-pro-preview`. xAI examples use `grok-3-latest` / `grok-4-latest`. Line lists per file:
   - comprehensive (175): 154, 181, 185, 198, 205, 229, 271, 309, 374–398, 419–420, 433, 461, 490, 534, 567, 678 … 6456;
   - model_providers (41): 17, 24, 27, 60, 63, 79, 114, 129, 174, 179, 207–209, 258, 259 …;
   - toolsets (31): 26, 100, 110, 134, 163, 187 …;
   - builtin_tools (29): 18, 23, 35, 47, 82, 100 …;
   - advanced_error_testing (26): 50, 75, 159, 181, 218 …, including 343 `google:gemini-2.0-flash`;
   - output_types (23): 22, 60, 115, 159, 193 …;
   - advanced_patterns (21);
   - direct_api (18);
   - recipes (14): 70, 227, 264 …;
   - concurrency (10);
   - hooks (10);
   - message_history (9);
   - streaming_server (8): 36, 74, 115, 180, 227, 271, 310, 419;
   - production (7): 66, 82, 309, 509, 623, 716, 801;
   - streaming (6): 23, 55, 173, 258, 280, 306;
   - integrations (5): 70, 80, 84, 126, 146;
   - durable (4): 32, 67, 137, 189;
   - diagrams (3): 16, 46 (`claude-3-5-sonnet`), 603;
   - embeddings (3): 50, 314, 549;
   - evals (3): 124, 131, 246;
   - testing (2): 18, 54;
   - index.mdx: 138.
   
   The full list was generated by grep. **Fix:** pick one current default per provider and replace globally: `anthropic:claude-sonnet-5-5`, `anthropic:claude-haiku-5-5` for the cheap tier, `openai:gpt-6-sol` (check OpenAI's live list), `google:gemini-3.5-flash`. Keep historical mentions only in the "Upgrading" and "What's new" sections.
2. **comprehensive_guide.md:2188–2190.** "`'early'` (default) stops the moment a final result is available…" is false: the default became `'graceful'` in 2.0. The guide's own upgrade table (l.1730) and output_types.md:135 say so. The same paragraph says `end_strategy` can be passed to `agent.run(...)`, but `run` has no such parameter. **Fix:** rewrite the paragraph: `'graceful'` is the default, `'early'` is opt-in, and the setting is constructor-only.
3. **comprehensive_guide.md:1749–1769.** The `Agent` constructor stub shows `end_strategy: EndStrategy = 'early'` and `deps_type = NoneType`. **Fix:** `'graceful'` and `object`.
4. **comprehensive_guide.md:1569–1571 vs builtin_tools.md:61–63.** The native-tool provider tables contradict each other and the source. In 2.59.0, `WebSearchTool` is supported by Anthropic, OpenAI Responses, Google, Groq, xAI and OpenRouter, not Bedrock. `CodeExecutionTool` is supported by Anthropic, OpenAI Responses, Google, Bedrock (Nova 2.0) and xAI. **Fix:** keep one table, in builtin_tools.md, copied from the class docstrings, and link to it from the comprehensive guide.
5. **comprehensive_guide.md:5473–5577.** The deprecated `TemporalAgent` / `DBOSAgent` / `PrefectAgent` get full worked examples (l.5492, 5509, 5571), including Temporal and Prefect wrappers with no capability alternative shown. **Fix:** cut this to a short "deprecated wrappers" note and link to durable_execution.md, which already has the `*Durability` capability examples.
6. **Hard-coded source citations are all wrong (16 places).** For example, testing.md:30/485 `models/test.py:60` (the class is at l.62), testing.md:240/487 `agent/__init__.py:1639` (`override` is at l.2221), message_history.md:71 / testing.md:489 `messages.py:2034` (the adapter is at l.3884), `_agent_graph.py:1791` (`capture_run_messages` is at l.3180), output_types.md:626 `:1911`, toolsets.md:560 `:2237`, testing.md:87 `function.py:45`. **Fix:** drop the line numbers and cite module and symbol only.
7. **Sidebar (astro.config.mjs:196).** PydanticAI has a single "Overview" link, so 22 pages are reachable only from index cards. **Fix:** add a collapsed group with Overview, Comprehensive guide, the topic pages, Recipes, Production and Diagrams, as LangGraph has.
8. **pydantic_ai_direct_api.md is an orphan.** No page links to it (grep for `direct_api` finds only old update notes). **Fix:** add it to the index Jump-to-topic and Reference grids and to the sidebar.
9. **production_guide.md as a whole.** Generic content. Agents are rebuilt per request (l.63–92, 801), which contradicts comprehensive l.41 "instantiated once and reused". The healthcheck uses `requests`, which is not installed (l.162). `/health` and `/ready` are never defined. The cache code calls a sync Redis client in async functions and an undefined `logger` (l.691). The page covers none of the following: `FallbackModel`, `ConcurrencyLimiter` / `limit_model_concurrency`, transport retries (`RetryConfig`, `AsyncTenacityTransport`), the `Instrumentation` capability or `Agent.instrument_all()`, `UsageLimits(cost_limit=…)` with `RunUsage.cost`, tool timeouts, `ApprovalRequiredToolset`, SSRF-safe `web_fetch_tool`, prompt-injection handling for MCP or tool output, the Pydantic AI Gateway, or testing in CI. **Fix:** rewrite it around these framework features, and link to durable_execution.md and evals for operational guidance.
10. **production_guide.md:740–778.** An nginx config sits inside a `python` block. **Fix:** retag it as `nginx`.
11. **pydantic_ai_comprehensive_guide.md is 6,476 lines.** Lines 1738–6476 are the former class deep-dives pasted in, including private and internal APIs: `MessagesBuilder` from `pydantic_ai.ui._messages_builder` (l.2365), `SystemPromptRunner` (l.2387), `_LazyAsyncClient` (l.2838), `_without_builtin_tools`, `_infer_base_url` / `pylf_v*` keys (l.3082), `_RequestParams`, and Prefect's `call_tool` implementation (l.5556–5568). **Fix:** cut the reference to public API only, move worked examples to their topic pages, and link to the upstream API docs for the rest. Aim for under 1,500 lines.
12. **toolsets.md duplicated sections.**
   - `PrefixedToolset`: l.~106, "collision-free namespace isolation" (~l.420), and "Toolset wrappers in detail" (~l.1180).
   - `FilteredToolset`: three times.
   - `DeferredLoadingToolset`: three times.
   - `ApprovalRequiredToolset`: twice.
   - `PreparedToolset`: twice.
   
   **Fix:** keep one section per wrapper and merge the examples.
13. **comprehensive_guide.md:1700.** "Observability & Logfire → Integrations", but the integrations page has no Logfire content; observability is spread across production_guide.md:281, model_providers.md (InstrumentationSettings) and comprehensive l.5021. **Fix:** point the row to the production guide's Observability section and consolidate the material there.
14. **comprehensive_guide.md:1063–1144.** Blocks l.1063, 1106 and 1144 use `Agent` without importing it. l.1064 imports the Pydantic v1 `validator`, and l.806 uses the deprecated `min_items=`. **Fix:** add `from pydantic_ai import Agent`, drop `validator`, and use `min_length=1`.

### Minor

1. **comprehensive_guide.md:82 and model_providers.md:86.** The default bundle is said to include `retries`. The actual dependency is `pydantic-ai-slim[anthropic,cli,evals,google,logfire,mcp,openai,web]`. **Fix:** remove "retries" (or add "install `[retries]` separately").
2. **model_providers.md:22–27.** "Same thing, constructed explicitly" uses `OpenAIChatModel`, but `openai:` means `OpenAIResponsesModel` (the page's own table, l.39). **Fix:** use `OpenAIResponsesModel` from `pydantic_ai.models.openai`.
3. **model_providers.md:88.** "KnownModelName literals (200+ entries)": 2.59.0 has 766. **Fix:** say "several hundred".
4. **toolsets.md:~110 ("Tool-name collisions raise at construction time").** `CombinedToolset([a, b])` with duplicate names constructs fine; the error comes when tools are listed during a run. **Fix:** say "raise when the run lists tools".
5. **toolsets.md:61 FunctionToolset table.** `max_retries` default is listed as `1`; the real default is `None` (inherits the agent's). **Fix:** correct the default.
6. **comprehensive_guide.md:2191.** "`AgentRetries = int | None`": it is a `TypedDict` (`retries={'tool': N, 'output': M}`), as output_types.md:339 says. **Fix:** align with that.
7. **9 pages lack `language: python` frontmatter:** advanced_patterns, comprehensive_guide, diagrams, durable_execution, evals_2025, graph_support, integrations_2025, production_guide, recipes. **Fix:** add `language: python`.
8. **Title naming is inconsistent.** "PydanticAI: …" (hooks, toolsets, testing, …) vs "Pydantic AI: …" (comprehensive, recipes, production, …). **Fix:** use the upstream spelling "Pydantic AI" everywhere.
9. **Editorial residue in page bodies:**
   - comprehensive l.2157 and l.3837 "**Correction:** …";
   - comprehensive l.3886–3887 "confirmed fully removed — grep found zero trace";
   - comprehensive l.5592;
   - durable l.9, graph l.9, evals l.9, integrations l.9: "This page was rewritten for V2. Earlier editions described…".
   
   **Fix:** state the current API plainly and move the history to the index revision table.
10. **Dated slugs** `pydantic_ai_evals_2025` and `pydantic_ai_integrations_2025`. **Fix:** rename to `pydantic_ai_evals` / `pydantic_ai_integrations`, with redirects in astro.config.mjs.
11. **recipes.md:351–400 and advanced_patterns.md:270–282.** These use the raw `openai.AsyncOpenAI().embeddings.create` although the guide has an Embeddings page. **Fix:** use `Embedder('openai:text-embedding-3-small').embed_query(...)`.
12. **Unverifiable marketing fluff.**
   - recipes.md:3 and :9 "production-tested" — **fix:** delete it.
   - production_guide.md:3 "Enterprise-grade" and advanced_patterns "Expert-level" — **fix:** tone these down.
   - The index "Every page in this guide is verified against this release" is contradicted by the failures above — **fix:** change it to "checked against" once the blockers are fixed.
13. **index.mdx "What's new" row "1.84–1.88: capabilities system introduced (… `ModelProfile`)".** `ModelProfile` predates 1.0. **Fix:** remove `ModelProfile` from that row.
14. **comprehensive_guide.md:126–135.** A `.env` file is in a `python` block, and l.137 imports `dotenv` (not a dependency). **Fix:** retag as `bash`/`ini` and note `pip install python-dotenv`.
15. **Small slips:**
   - comprehensive_guide.md:1194 typo `calculate_factororial`;
   - l.1111 shadows `pydantic_ai.TextOutput` with a user class;
   - l.944 `Annotated[Union[...], 'response']` is labelled a discriminated union (it is not);
   - l.1463 the prepare function is annotated `-> Tool | None` (should be `ToolDefinition | None`).
   
   **Fix:** correct each.
16. **production_guide.md:305–306.** `logfire.instrument_sqlalchemy()` / `instrument_asyncpg()` fail without `logfire[sqlalchemy,asyncpg]`. **Fix:** add the install line or drop the calls. Also: testing.md:389 says TestModel "randomises tool args" while l.307 says it is deterministic (it is seeded, default `seed=0`); make the two consistent.

---

#### Top 5 improvements

1. **Fix the 13 blockers and add a CI check.** Run every non-stub block against `TestModel` (the harness in `scratchpad/assess-pydanticai/` takes about 10 minutes). Retag the 167 pseudo-signature listings as `text` so they stop posing as code.
2. **Do one global model-ID refresh** to the current Claude 5 / GPT-6 / Gemini 3.x names, with one canonical default per provider. This touches about 490 lines, but it is mechanical.
3. **Split and trim the comprehensive guide.**
   - Keep the tutorial (l.1–1736) as the "Getting started" page.
   - Cut the class reference to public API only (drop private internals, deprecated-wrapper examples and editorial notes), or move it into the topic pages.
   - Fix the false reference claims (end strategy, `AgentRetries`, provider tables, `InstrumentationSettings`).
4. **Rebuild navigation.** Add a full sidebar group, link the orphan Direct API page, de-duplicate the toolsets page, rename the `_2025` slugs, and complete the frontmatter (`language`) and title spelling.
5. **Rewrite the production guide around Pydantic AI's own features:**
   - `FallbackModel`, `ConcurrencyLimiter` and transport retries;
   - the `Instrumentation` capability and Logfire, with one observability section the other pages link to;
   - `UsageLimits(cost_limit)` with `RunUsage.cost`;
   - tool timeouts, approvals and SSRF-safe fetching;
   - durability capabilities and evals in CI.
   
   Remove the generic K8s/nginx filler and the trailing stub.

---

### Assessment: OpenAI Agents SDK guides (Python + TypeScript)

Scope: `src/content/docs/openai-agents-sdk-guides/` (10 pages, Python) and `src/content/docs/openai-agents-sdk-typescript-guide/` (9 pages, TypeScript). Read in reading order (index, comprehensive, features, middleware/advanced, recipes, production, Swarm migration, streaming server, diagrams). Assessed 2026-10-21 against `openai-agents` 0.24.0 and `@openai/agents` 0.21.0 (both match `src/data/versions.json`, and both are the latest on PyPI/npm today).

Scratch work: `/tmp/claude-0/-home-user-AgentGuides/55b0d51f-358a-5617-8094-250bd4df3d90/scratchpad/assess-openai-agents/` (harness `harness.py`, `runall.sh`, results `run_py_*.txt`, `ts/run_ts.txt`, `ts/tsc2.log`, `ts/raw/tsc_raw.log`).

#### Scorecard

| # | Criterion | Score | Justification |
|---|-----------|-------|---------------|
| 1 | Correctness of code | 3/5 | Core SDK calls are now mostly right (imports, kwargs and signatures check out), but 6 Python blocks crash on construction (required `tool_config`/`vector_store_ids`, abstract classes, non-strict output types), two TS tools throw at definition, and the production pages have runtime bugs (`asyncio.time()`, `result.usage`, a guardrail that blocks "should"). Under `tsc --strict`, 62 of 119 TS blocks fail as written. |
| 2 | Currency | 3/5 | Versions use placeholders and `check-versions` is clean. OpenAI model IDs are current (`gpt-5.6-sol/luna`, `gpt-realtime-2.1`). But there are 16 stale third-party IDs (`claude-sonnet-4-5`, `gemini-2.5-flash`), a Langfuse API that no longer exists, a deprecated OTel package, and a GitHub MCP URL that is wrong. |
| 3 | Accuracy of prose | 2/5 | The changelog tables are accurate. Elsewhere there are invented features: traces "automatically appear in the OpenAI evaluation dashboard" and are "automatically logged for fine-tuning". Also impossible dates ("first written January 2025 for v0.2.9"), a "90% SimpleQA" heading, "battle-tested / better performance" filler, and section titles the code doesn't deliver ("Message Filtering", "Streaming Guardrails", "Non-Strict Output Types", "Voice Agents"). |
| 4 | Coverage | 2/5 | Python leaves out human-in-the-loop approvals, lifecycle hooks, tool guardrails, Realtime/Voice, hosted MCP, handoff input filters, the testing utilities (`agents.testing`), Runner-managed retries, Temporal/Dapr and Encrypted/compaction sessions. TS has no `agent.asTool()`, almost nothing on sandbox agents, no programmatic tool calling and no lifecycle events. Padding: three overlapping version/changelog sections per language, and Swarm-migration pages of 900+ lines. |
| 5 | Structure & navigation | 3/5 | Each index gives install, a working minimal example and a full map, with no orphans and no broken links/anchors. But "Zero → Hero" step 9 (Python) points to a 24-line stub. The "2025 features" pages duplicate the comprehensive guides. One "Core Primitives Deep Dive" heading is left over, and TS Realtime/breaking-changes sit under "Summary". |
| 6 | Consistency | 3/5 | The two languages use different `framework:` values (`openai-agents-sdk` vs `openai-agents-sdk-typescript`). `language:` is missing on 16 of 19 pages. Python's Git MCP server is wrong while TS's is right. Sandbox backend lists differ. Emoji ticks appear in 5 pages, and there are a few American spellings. |
| 7 | Production readiness | 2/5 | Mostly generic patterns (hand-rolled pools, caches, circuit breakers, secret managers with empty Vault branches). The SDK's own production controls are missing: `trace_include_sensitive_data`, `ModelSettings.retry`/`timeout`, `signal`, tracing processors, scripted-model tests. The unit tests call the live API. CI `docker login` is broken, and the PromQL/OTel setup is wrong. |

**Verdict.** The SDK surface has been cleaned up well: nearly every import and keyword argument matches 0.24.0 / 0.21.0, and both quick-starts run end to end with the SDK's own scripted model. What lets the guides down is the material around that core: a handful of hosted-tool and output-type examples that crash, production pages full of generic or subtly broken code, invented claims about evaluation and fine-tuning, and large gaps in coverage (HITL, hooks, Realtime/Voice and testing in Python; `asTool` and sandbox in TS). It needs one focused pass on the production pages and the "features" pages before it is a reliable reference.

#### Code verification results

Method:
- **Python.** I extracted every fenced block and compiled it (top-level `await` allowed). I checked every `agents.*` import and every keyword argument against the installed 0.24.0 signatures. Then I **executed** each block in its own process, with `MultiProvider.get_model` patched to the SDK's `agents.testing.ScriptedModel`. That model returns schema-valid dummy JSON for `output_type` agents and "ok" otherwise. Tracing was off, `asyncio.sleep` was a no-op and there was a 20 s cap.
- **TypeScript.** I extracted every block and ran `tsc --strict` twice: once as written, and once with the missing `@openai/agents`/`zod` imports injected, so that real API errors could be told apart from snippet fragments. I then emitted JS and executed every block with `setDefaultModelProvider()` returning `ScriptedModel` from `@openai/agents/testing`.

"Env" means it failed only because a service or key was absent (Redis, Postgres, an MCP server, `npx`, an API key). "Fragment" means a deliberate partial snippet or Swarm pseudo-code.

### Python (`openai-agents` 0.24.0)

| Page | Blocks | Ran clean | Genuine failures | Env | Fragment/pseudo |
|------|--------|-----------|------------------|-----|-----------------|
| index.mdx | 1 | 1 | 0 | 0 | 0 |
| comprehensive_guide | 77 | 60 | 9 (L1318, L1342, L1369, L1615, L1660 URL, L1735 package, L1762, L2223, L2734) | 6 | 4 |
| 2025_features | 29 | 20 | 3 (L650 package, L676 URL, plus false claims L233–284/L693–744) | 0 | 8 |
| middleware_python | 1 | 1 | 0 | 0 | 0 |
| advanced_multi_agent | 2 | 2 | 0 | 0 | 0 |
| recipes | 13 | 10 | 3 (L246, L395 logic, L963) | 0 | 0 |
| production_guide | 20 | 19 | 5 (L833, L915, L1305 CI, L1438, L1536; not reached by the harness) | 1 | 0 |
| swarm_migration | 31 | 10 | 1 (L817 missing `import os`) | 1 | 19 (Swarm "before" code, fragments) |
| streaming_server_fastapi | 2 | 2 | 0 | 0 | 0 |
| diagrams | 0 | – | – | – | – |
| **Total** | **176** | **125** | **21** | **8** | **31** |

Both quick-starts (index L104, comprehensive L578) ran end to end with the scripted model. The static checks found no hallucinated `agents.*` imports and no bad keyword arguments (the earlier `trace.export` flag was a false positive caused by a shadowed parameter).

### TypeScript (`@openai/agents` 0.21.0)

| Page | Blocks | `tsc --strict` as written: blocks with errors | Real type/API errors (imports injected, non-strict) | Executed OK (scripted model) |
|------|--------|-------------------------|---------------------------------------|------------------------------|
| index.mdx | 1 | 0 | 0 | 1/1 |
| comprehensive_guide | 44 | 32 | 5 (L1465, L1515 runtime, L1776, L1899, L2302) | 38/44 |
| 2025_features | 20 | 6 (+1 pseudo-code syntax error L47) | 0 | 14/19 |
| middleware_typescript | 2 | 1 | 0 | 1/2 (fragment) |
| recipes | 18 | 4 | 0 | 16/18 (2 long-running servers/cron) |
| production_guide | 14 | 9 | 4 (L818, L881, L927, L1044) | 6/14 |
| swarm_migration | 20 | 10 (+1 pseudo syntax L570) | 1 (L688) + Swarm pseudo-code | 8/19 |
| streaming_server_express | 2 | 0 | 0 | 0/2 (servers; they listen correctly) |
| diagrams | 0 | – | – | – |
| **Total** | **121** | **62 of 119 fail strict as written** | **10 genuine** | **84/119** |

Breakdown of the 350 strict errors as written:
- 241 are "cannot find name" (`z` 88, `Agent` 59, `run` 33, `tool` 17): continuation snippets with no imports.
- About 45 are strict-null or `unknown` catch-variable errors: `finalOutput` possibly `undefined` returned as `string`, or `error.message` on `unknown`.
- About 10 are genuine API or type errors, listed below.

`index.mdx` L94 runs end to end.

#### Issues

### Blocker

| # | File:line | Problem | Fix |
|---|-----------|---------|-----|
| B1 | py `openai_agents_sdk_comprehensive_guide.md:1325` | `CodeInterpreterTool()` raises `TypeError: missing 'tool_config'` | `CodeInterpreterTool(tool_config={"type": "code_interpreter", "container": {"type": "auto"}})`, and fix the note at L1316 that says the config is optional |
| B2 | py `comprehensive_guide.md:1376` | `ImageGenerationTool()` raises `TypeError: missing 'tool_config'` | `ImageGenerationTool(tool_config={"type": "image_generation", "quality": "low"})` |
| B3 | py `comprehensive_guide.md:1611-1639` | "Non-Strict Output Types" never turns strict off; `data: Any` / `metadata: dict` raise `UserError: Strict JSON schema is enabled…` | `output_type=AgentOutputSchema(FlexibleResponse, strict_json_schema=False)` |
| B4 | py `comprehensive_guide.md:1762-1795` | `CustomToolServer(MCPServer)` only defines `handle_call_tool`; instantiating it raises `TypeError: Can't instantiate abstract class` (7 abstract methods) | Replace with a real FastMCP server (`mcp.server.fastmcp.FastMCP`) run over stdio and attached with `MCPServerStdio`, or implement `connect/cleanup/name/list_tools/call_tool/list_prompts/get_prompt` |
| B5 | py `openai_agents_sdk_recipes.md:246` | `FileSearchTool(max_num_results=5)` raises `TypeError: missing 'vector_store_ids'` | Add `vector_store_ids=["vs_..."]` |
| B6 | py `recipes.md:963-969` | `ContractAnalysis.key_terms: dict` used as `output_type` raises a strict-schema `UserError` | `key_terms: list[KeyTerm]` (name/value model) or `AgentOutputSchema(..., strict_json_schema=False)` |
| B7 | py `openai_agents_sdk_production_guide.md:1536` | `asyncio.time()` does not exist; every request in the "Customer Service Platform" returns HTTP 500 | `int(time.time())` with `import time` |
| B8 | py `openai_agents_sdk_2025_features.md:233-284, 693-744` | Invented features: "Traces automatically appear in OpenAI evaluation dashboard", "Data automatically logged for fine-tuning", "Train student model on teacher outputs". Tracing does none of this. | Delete the "Fine-tuning Integration", "Distillation Tools" and "Evaluation and Fine-tuning Integration" sections, or replace them with a real export (a custom `TracingProcessor` writing JSONL for the Evals API) |
| B9 | ts `openai_agents_sdk_typescript_comprehensive_guide.md:1517-1557` | `z.string().pipe(z.coerce.date())` throws `UserError: Date cannot be represented in JSON Schema` when `tool()` is called; the `z.discriminatedUnion` root throws "must produce … `type: "object"` at the root" | Use `z.string().describe('ISO date')` and parse inside `execute`; wrap the union: `z.object({ query: z.discriminatedUnion(...) })` |
| B10 | py `comprehensive_guide.md:2223` | `await session.get_items()[-1]` subscripts a coroutine (`TypeError` whenever history is non-empty; it passes only because `items` is empty) | `last_item = (await session.get_items(limit=1) or [None])[-1]` |

### Major

| # | File:line | Problem | Fix |
|---|-----------|---------|-----|
| M1 | py `comprehensive_guide.md:1660`, `2025_features.md:676` | GitHub MCP URL `https://api.github.com/mcp` does not exist | `https://api.githubcopilot.com/mcp/` |
| M2 | py `comprehensive_guide.md:1738`, `2025_features.md:650` | `npx @modelcontextprotocol/server-git` returns 404 on npm (the Git server is Python) | `"command": "uvx", "args": ["mcp-server-git", "--repository", str(repo_path)]` (as the TS page does at L689) |
| M3 | py `comprehensive_guide.md:1342-1365` | `ComputerTool(computer=MyBrowser())` with an empty `AsyncComputer` subclass raises `TypeError` (9 abstract methods) | Show a minimal Playwright implementation, or mark the block clearly as an outline and do not call it |
| M4 | py `comprehensive_guide.md:2736` | `from langfuse.decorators import observe` was removed in Langfuse v3, and `@observe` does not capture SDK spans anyway | Use Langfuse's OpenAI Agents integration (OpenInference `OpenAIAgentsInstrumentor` plus OTLP), or `add_trace_processor` with a Langfuse exporter |
| M5 | py `production_guide.md:830-833` | `getattr(result, "usage", {}).get("total_tokens", 0)`: `RunResult` has no `usage`, so the token metric is always 0; `model=agent.model` is `None` for default-model agents | `result.context_wrapper.usage.total_tokens`; label with `agent.model or "default"` |
| M6 | py `production_guide.md:914-923` | The "injection_prevention" guardrail trips on `(`, `$`, `;` and any "sh" substring ("Should I share…" and "price (in GBP)" are blocked; verified) | Remove the shell-metacharacter and `bash/sh` patterns; use a model-based guardrail agent, as in the upstream docs |
| M7 | py `production_guide.md:1434-1438` | `datetime` not imported (`NameError`); the session id includes a timestamp, so every call starts a new conversation; `api_key` is stored in run context | Import `datetime`; use `f"{tenant_id}:{user_id}"`; keep keys out of context |
| M8 | py `production_guide.md:1305` | `docker login … -u $ -p -` is invalid; `actions/checkout@v3` and `setup-python@v4` are outdated | `echo "$TOKEN" \| docker login ghcr.io -u ${{ github.actor }} --password-stdin`; use `@v5` actions |
| M9 | py `production_guide.md:1336-1373` | "SQLAlchemy Session Backend" builds an unrelated ORM table (`metadata = Column(...)` triggers a SQLAlchemy reserved-name warning) and never connects to the SDK | Replace with `SQLAlchemySession.from_url(..., create_tables=True)` (already correct at comprehensive L2275) |
| M10 | py `production_guide.md:1110-1231`, `comprehensive_guide.md:2995-3027` | Unit tests hit the live API and assert on model text ("4" in output) | Use `agents.testing.ScriptedModel` with `assistant_message`/`function_call` (added 0.21) for deterministic tests |
| M11 | py `recipes.md:376-395` | `portfolio_manager` uses handoffs, but the run can end on `collector_agent` (str), so `final_output_as(StockAnalysis)` returns a str and `.ticker` raises `AttributeError` (reproduced) | Use agents-as-tools on a manager with `output_type=StockAnalysis` |
| M12 | py `openai_agents_sdk_middleware_python.md:1-24` | A 24-line stub with one framework-agnostic snippet, yet index L44 sells it as "Lifecycle hooks, custom processing · 20 min" | Write it properly: `RunHooks`/`AgentHooks`, `@tool_input_guardrail`/`@tool_output_guardrail`, `call_model_input_filter`, `needs_approval` |
| M13 | py (all pages) | Missing core features: HITL (`needs_approval`, `result.interruptions`, `RunState` resume), lifecycle hooks, tool guardrails, `RealtimeAgent`/`RealtimeRunner`, `VoicePipeline`, `HostedMCPTool`, `MCPServerSse`, handoff `input_filter`, `ShellTool`/`ApplyPatchTool`, `EncryptedSession`/compaction sessions, Runner-managed retries (`ModelSettings.retry`), `[temporal]`/`[dapr]`/`[any-llm]` extras, `draw_graph` | Add a section for each to the comprehensive guide; the "Realtime Experiences" section (L2782-2833) is a `input()` loop and a fake "voice" demo and should be replaced with a real `RealtimeAgent` and `VoicePipeline` example |
| M14 | py `comprehensive_guide.md:921-968` | "Agent Delegation with Message Filtering" has no filter (it duplicates the L374 example) | Show `handoff(agent, input_filter=handoff_filters.remove_all_tools)` |
| M15 | py `comprehensive_guide.md:2123-2161`; `diagrams.md:44-50` | "Streaming Guardrails" uses an output guardrail, which runs only after the final output. The diagram shows input guardrails before the model call, but `run_in_parallel=True` is the default. | Retitle the section; explain `@input_guardrail(run_in_parallel=False)` for blocking checks |
| M16 | py `production_guide.md` (whole page) | No SDK-specific production controls: `RunConfig(trace_include_sensitive_data=False)`, `tracing_disabled`/ZDR, `ModelSettings(retry=…, timeout=…)`, tool `timeout`, prompt caching, Temporal durable runs | Add a short "SDK controls" section before the generic patterns; cut the unrelated connection-pool and round-robin client (L425-507) |
| M17 | ts `comprehensive_guide.md:1197-1252` | "Agent as Tools Pattern" wraps `run()` in hand-made `tool()`s; the built-in `agent.asTool()` is never shown in the TS guide | Use `summariserAgent.asTool({ toolName, toolDescription })` |
| M18 | ts `comprehensive_guide.md:389-428`, `2302-2365` | `createTypedAgent` ignores `outputSchema`/`onComplete`; `TypedAgentFactory` builds an agent without `outputType` and `JSON.parse`s free text | Pass `outputType: schema` and read the typed `result.finalOutput`; delete the dead factory |
| M19 | ts `comprehensive_guide.md:1465-1495, 1776, 1899`; `swarm_migration_guide.md:688` | Generic `tool()` wrappers (`ToolBuilder`, etc.) do not type-check even with `strict:false` (TS2322/TS2345 on `parameters`/`execute`) | Drop the wrappers or constrain them: `P extends z.ZodObject<z.ZodRawShape>` and pass the result of `tool<P>()` |
| M20 | ts `production_guide.md:927-938` | `PrometheusExporter` passed to `PeriodicExportingMetricReader` (it *is* a reader; TS2741); `MeterProvider` not imported | `new MeterProvider({ readers: [new PrometheusExporter({ port: 9464 })] })` with `import { MeterProvider } from '@opentelemetry/sdk-metrics'` |
| M21 | ts `production_guide.md:881-891` | `@opentelemetry/node` is deprecated (renamed); `@opentelemetry/exporter-jaeger` is unsupported; `BatchSpanProcessor`/`SpanStatusCode` not imported | `@opentelemetry/sdk-trace-node` plus `@opentelemetry/exporter-trace-otlp-http`; or show `addTraceProcessor()` from the SDK |
| M22 | ts `production_guide.md:818-829` | `xss(..., { stripIgnoredTag: true })` is a typo (TS2561); the option is silently ignored in a security sanitiser | `stripIgnoreTag: true` |
| M23 | ts `production_guide.md:1044-1059` | `import { Queue } from 'bull'` is type-only (TS2693); runtime `ReferenceError` | `import Queue from 'bull'` (or BullMQ `import { Queue, Worker } from 'bullmq'`) |
| M24 | ts `production_guide.md:1183-1214` | Jest tests call the live API; the "should respect timeout" test expects `run()` to take longer than 5 s | Use `ScriptedModel` from `@openai/agents/testing`; test timeouts with `run(..., { signal: AbortSignal.timeout(…) })` |
| M25 | ts (all pages) | Thin coverage: sandbox agents (one sentence at 2025_features L978), programmatic tool calling (TS 0.14), agent/runner lifecycle events (`agent.on('agent_start')`, `runner.on`), `codeInterpreterTool`/`computerTool`, `MCPServerSSE`, tracing processors | Add sections to the comprehensive guide; fold the Realtime section (L2381) into its own `##` |
| M26 | py `comprehensive_guide.md:2859, 2866, 2873`; `2025_features.md:375, 382, 389`; `swarm_migration_guide.md:156, 163, 170`; ts `comprehensive_guide.md:348`; `2025_features.md:774, 780`; `swarm_migration_guide.md:189, 196` | Stale model IDs: `claude-sonnet-4-5` (6), `gemini-2.5-flash` (5), `ollama/llama3.1` (3) | `claude-sonnet-5-5`, `gemini-3.5-flash` (as used elsewhere on the site), a current Ollama tag |
| M27 | py `2025_features.md` and ts `typescript_2025_features.md` (whole pages) | Year-titled "Features and Updates (2025–2026)" pages repeat the comprehensive guide (primitives, guardrails, sessions, MCP, cost) and add a third version table, next to the index "What's new" and the comprehensive "Breaking changes" | Move the HITL/PTC/WebSocket content into the comprehensive guides, keep one version table per language (index), and redirect the rest |
| M28 | py `comprehensive_guide.md:2617` | Heading "Built-in Web Search (90% SimpleQA Accuracy)" is a 2025 launch figure for a different model | Drop the number from the heading |

### Minor

| # | File:line | Problem | Fix |
|---|-----------|---------|-----|
| m1 | py `2025_features.md:7`, `swarm_migration_guide.md:897`; ts `typescript_2025_features.md:7`, `typescript_swarm_migration_guide.md:7, 925` | "First written January 2025 for v0.2.9" is impossible (the SDK launched in March 2025; 0.2.9 is from late summer 2025). The TS footer says "Last Updated: January 2025 / Agents SDK Version: Latest" | Use the revision-history dates (Nov 2025 / 0.6.1, 0.3.2); delete the TS footer |
| m2 | ts `typescript_2025_features.md:967` | "0.8 — Human-in-the-loop approvals … Zod 4": HITL predates 0.1, and Zod v4 became required in 0.4.0 (changelog) | "0.4 — Zod v4 required"; drop the HITL claim |
| m3 | 16 of 19 non-index pages | `language:` frontmatter missing; TS pages use `framework: openai-agents-sdk-typescript` while the TS index uses `openai-agents-sdk` | Add `language: python/typescript`; use one framework id |
| m4 | ts `typescript_comprehensive_guide.md:102` | "Core Primitives Deep Dive" heading (banned framing) | "Core primitives" |
| m5 | ts `typescript_comprehensive_guide.md:2367-2428` | "Realtime Voice Agents" and "Breaking changes by version" sit under `## Summary` and are missing from the TOC | Promote them to `##`, before the Summary |
| m6 | py `production_guide.md:3` | Description truncated mid-word ("performance optim") | Rewrite to fit about 160 characters |
| m7 | py `index.mdx:3` | Description is meta ("Zero to hero with chaptered checkpoints and topic-jump anchors") | Describe the SDK |
| m8 | ts `openai_agents_streaming_server_express.md:69, 82, 107` | Second server listens on 8081, but Docker/K8s expose 8080; `CMD ["node","server.js"]` runs TS source with no build step | Use one port; add `npm run build` and `dist/server.js` |
| m9 | ts `typescript_comprehensive_guide.md:1146-1195` | `Promise.race` timeout timers are never cleared, so the process hangs for up to 30 s after finishing (observed) | Use `run(agent, input, { signal: AbortSignal.timeout(ms) })` |
| m10 | ts many blocks (845, 913, 2040, 2083, 2170, recipes 125/695/735, production 555/881) | `result.finalOutput` returned as `string` and `error.message` on `unknown` fail `strict:true`, which the guide's own tsconfig (L63) enables | `result.finalOutput ?? ''`; `error instanceof Error ? error.message : String(error)` |
| m11 | ts `typescript_diagrams.md:206, 372` | Diagram lists "Redis (Sessions)"; the TS SDK ships no Redis session. Python `diagrams.md` / ts L198 show invented "Session End: archive history, save statistics" | Show `MemorySession` / `OpenAIConversationsSession` / custom `Session`; remove the invented lifecycle step |
| m12 | py `comprehensive_guide.md:60` vs `2025_features.md:819` | Sandbox backend lists differ (Runloop missing in one; it is still a Python extra) | Use one list |
| m13 | py `comprehensive_guide.md:2587` | `Reasoning(effort="minimal")` was GPT-5-only; the default for `gpt-5.6-*` is `"none"` | Use `"none"` or `"low"` |
| m14 | py `swarm_migration_guide.md:44-50, 842-862`; ts `swarm_migration_guide.md:37-56` | Marketing filler ("battle-tested", "Better Performance: optimised execution", "Stable API" despite 0.Y breaking changes), invented migration timelines; the TS page lists "Swarm limitations" for a TS Swarm it says does not exist | Cut these; state the 0.Y policy plainly |
| m15 | py `2025_features.md:186-201` | Snippet mixes `Runner.run_sync` and top-level `await` (fails inside a running loop) | Split it into sync and async examples |
| m16 | py `swarm_migration_guide.md:817` | `os` used without import | Add `import os` |
| m17 | 5 pages (py comprehensive 7, 2025_features 14, swarm 7; ts 2025_features 14, swarm 12) | Emoji ticks/icons in prose and print strings | Plain text |
| m18 | py `comprehensive_guide.md:2651, 3030`; `2025_features.md:21, 205, 209, 782`; ts several | American spellings (Visualization, Optimization) | British spellings |
| m19 | py `index.mdx` (35 links), ts `index.mdx` (27), plus 7 in other pages | Relative `./…` internal links; AGENTS.md asks for root-absolute | `/openai-agents-sdk-guides/...` |
| m20 | py `recipes.md:7`, `production_guide.md:1550`; ts `production_guide.md:1221` | "production-ready implementations", "enterprise-ready patterns" | Neutral wording |
| m21 | ts `typescript_comprehensive_guide.md:1033` | Imports `removeAllTools` from the transitive `@openai/agents-core/extensions` | Note that it needs `@openai/agents-core` installed explicitly, or filter by hand |

Checks that passed:
- `node scripts/check-versions.mjs`: no errors or warnings for either folder.
- `node scripts/check-links.mjs`: 0 broken links site-wide, and no broken intra-guide anchors (own scan).
- No stray `# ` H1s in page bodies.
- No TODO/TBD/"Vol." markers.
- All pages are reachable from their index.
- The sidebar has one entry per language, which is the site's convention.
- Changelog tables match the upstream `docs/release.md` (Python) and `packages/agents-core/CHANGELOG.md` (JS) for 0.13–0.22 / 0.13–0.21.

#### Top 5 improvements

1. **Fix the ten blockers and the hosted-tool examples.** Add the required `tool_config`/`vector_store_ids`, use `AgentOutputSchema(..., strict_json_schema=False)` for loose types, use a real MCP server example, a JSON-Schema-safe Zod tool, and `time.time()`. Delete the invented eval/fine-tuning sections. These are what a new reader copies first.
2. **Rewrite both production guides around the SDK's own controls.** Cover sensitive-data tracing flags, `ModelSettings.retry`/`timeout`, `signal`, tracing processors and OTel exporters, deterministic tests with `ScriptedModel` (both languages ship it), session backends that are actually the SDK's, and Temporal for durable runs. Drop the generic pool/cache/secret-manager code and fix the broken CI, metrics and guardrail snippets (M5–M9, M20–M24).
3. **Fill the Python coverage gaps.** Add HITL approvals with `RunState`, `RunHooks`/`AgentHooks`, tool guardrails, `RealtimeAgent` and `VoicePipeline`, `HostedMCPTool`, and handoff `input_filter`. Turn the middleware stub into the hooks/guardrails page the index promises. For TS, add `agent.asTool()`, sandbox agents, programmatic tool calling and lifecycle events.
4. **Merge the "2025 features" pages into the comprehensive guides and keep one version table per language** (on the index). Trim the two 900-line Swarm migration pages to a short side-by-side and checklist. This removes duplication and most of the marketing tone.
5. **Make every TS block pass `tsc --strict` with its own imports, and refresh the third-party model IDs and integrations.** Use `claude-sonnet-5-5`, `gemini-3.5-flash`, the Langfuse v3/OpenInference integration, `uvx mcp-server-git` and the correct GitHub MCP URL. Add `language:` frontmatter and one `framework` id across both folders.

---

### Assessment: AG2 (`ag2-guide/`) and Microsoft AutoGen / AG2 Classic (`autogen-guide/`)

Assessed 2026-10-21 against `ag2` 1.2.1, `autogen` 0.14.1 (AG2 Classic), `autogen-agentchat` / `autogen-core` / `autogen-ext` 0.7.5 (Microsoft AutoGen), NuGet `AutoGen` 0.2.3 (.NET). Scratch work: `/tmp/claude-0/-home-user-AgentGuides/55b0d51f-358a-5617-8094-250bd4df3d90/scratchpad/assess-ag2-autogen/`.

The guides cover three code bases, not two:

| Code base | Import | PyPI | Status | Where the site covers it |
|---|---|---|---|---|
| AG2 1.x | `import ag2` | `ag2` | Active | `ag2-guide/` (all pages) |
| AG2 Classic | `import autogen` | `autogen` (and `ag2<1.0`) | Maintenance (security/critical fixes) | `autogen-guide/python/` comprehensive, recipes, production, diagrams, migration; plus one section of `ag2-guide/ag2_comprehensive_guide.md` |
| Microsoft AutoGen 0.4–0.7 | `autogen_agentchat`, `autogen_core` | `autogen-agentchat` etc. (`pyautogen` is now a proxy for it) | Archived | `autogen-guide/python/` agentchat, advanced, observability; `autogen-guide/dotnet.md` |

#### Scorecards

### AG2 1.x guide (`ag2-guide/`)

| # | Criterion | Score | Justification |
|---|---|---|---|
| 1 | Correctness of code | 5 | 17/17 1.x blocks and 1/1 Classic block pass. All ran offline with scripted `TestClient`s, including Network, MCP (with a local MCP 2.x server and the npm filesystem server) and the pytest example (once `pytest-asyncio` is installed). |
| 2 | Currency | 3 | `check-versions` is clean, but there are 18× `gpt-4.1-mini` (two generations old), 2× `claude-sonnet-4-6` (should be `claude-sonnet-5-5`) and 1× `gemini-2.5-flash`. |
| 3 | Accuracy of prose | 4 | Release history, extras, signatures and A2A/MCP options all check out against the 1.2.1 source and PyPI. Three things are wrong: the "formerly AutoGen" title; provider-hosted `CodeExecutionTool` / `ShellTool` are presented as local code-executor replacements; and the production code-execution advice is muddled. |
| 4 | Coverage | 3 | Core features are solid (Agent, tools, structured output, HITL, streams, Network, MCP, A2A, testing). The harness (knowledge/assembly/compaction), middleware/observers, dependency injection, multimodal inputs, Tasks, Plugins and eval get one line or none. |
| 5 | Structure and navigation | 3 | The index has install, a working example and a page map, and there are no broken links. But the sidebar has a single "AG2" entry with no subpages, there is no home-page card, and site-level pages send AG2 readers to the Classic/Microsoft section. |
| 6 | Consistency | 4 | Same API style and terminology throughout, understated tone, British English. Four pages lack `language:` in frontmatter. |
| 7 | Production readiness | 3 | Current and specific (pinning, TestConfig, durable hub stores, multi-replica A2A, push URL validation), but no code (OTel setup, ASGI deployment) and the sandboxing advice is inaccurate. |

### AutoGen guide (`autogen-guide/`: Microsoft AutoGen + AG2 Classic)

| # | Criterion | Score | Justification |
|---|---|---|---|
| 1 | Correctness of code | 3 | Microsoft pages: 18/18 pass. Classic pages: 75/92 pass. 7 fail outright (bad kwargs, removed APIs, non-existent methods), 8 fail on any machine without a running Docker daemon, and 2 need Docker. The .NET snippet fails to compile as published (CS1061). |
| 2 | Currency | 2 | `check-versions` is clean, but the Classic pages use `gpt-4`, `gpt-3.5-turbo`, `claude-3-5-sonnet-20240620`, `gemini-pro` and `text-davinci-003`, plus the removed `langchain.llms` / `llama_index.GPTVectorStoreIndex` APIs and an `ag2[openai]==0.2.x` pin. |
| 3 | Accuracy of prose | 3 | Microsoft pages and timeline verified: `pyautogen` 0.10.0 proxies `autogen-agentchat`; 0.7.5 was released 2025-09-30; the upstream README says archived; the NuGet versions and dates match; and the "MCP adapter breaks on mcp 2.x" claim reproduces. On the Classic pages, swarm is said to live in `autogen.agentchat.contrib.swarm_agent`, but it was removed in 0.14.0. There is also a false OAI_CONFIG_LIST auto-discovery claim and "modern AutoGen" phrasing. |
| 4 | Coverage | 3 | The Microsoft core is well covered (AssistantAgent, teams, GraphFlow, MCP, state, Core, tracing). Memory, model context, Magentic-One, code executors and AutoGen Studio are missing. The Classic pages are padded: many near-identical prompt-only examples and filler production classes. |
| 5 | Structure and navigation | 2 | AG2 Classic sits under the sidebar label "Microsoft AutoGen (Python)". There is a placeholder stub in the comprehensive guide, migration content appears three times, a "Deep Dive" heading remains, and the sidebar has no subpage entries. |
| 6 | Consistency | 3 | The Classic pages mix `config_list` dicts with `LLMConfig`, and use "AG2" unqualified for Classic, marketing-style descriptions ("Enterprise-Grade"), some American spellings and exclamation marks. The Microsoft pages are consistent. |
| 7 | Production readiness | 2 | The Classic production guide is mostly generic filler: helper classes never wired in, async "examples" that only `sleep`, a fake LangChain integration, broken A2A deployment code. For an archived project the Microsoft observability page is adequate. |

### Are the projects clearly and correctly separated?

**In the prose, yes. In the site structure, no.**

What works:
- Every page has a banner saying which code base it covers.
- The AutoGen index timeline is accurate.
- The migration guide's "Which AutoGen do you have?" table (`autogen_to_ag2_migration_guide.md:12-21`) and paths A/B/C are correct and clear.
- The false "Microsoft rebranded AutoGen as AG2" narrative is gone.
- The package-name traps (`pyautogen` → Microsoft; `autogen` → Classic; `ag2>=1.0` → new framework) are stated correctly in several places.

What does not:
- AG2 Classic, an `ag2ai` project, is filed in the Microsoft guide's folder and under the sidebar label "Microsoft AutoGen (Python)" (`astro.config.mjs:117`). Five of the eight pages behind that label are AG2 Classic.
- Site-level pages still blur the projects:
  - `src/data/frameworks.ts:88-98` has one combined "AutoGen / AG2" card with the tagline "GroupChat and A2A" that links only to `/autogen-guide/`.
  - `guides.md:240-251` lists only `autogen-guide` pages under "AutoGen / AG2" and never links the AG2 1.x guide.
  - `frameworks.md:248-274` describes AG2 with Classic traits ("GroupChat for multi-agent", "conversation-based") but links to the 1.x guide.
- The AG2 index title is "AG2 (formerly AutoGen)" (`ag2-guide/index.mdx:2`), which implies a rename.
- Classic pages say "AG2" without qualification (`autogen_diagrams.md:31`, `:731`; `autogen_production_guide.md:455`, `:1026`), even though AG2 now means 1.x.

**Overall verdict.** The AG2 1.x guide is accurate and every example runs, but it is a compact reference that needs current model IDs, its own navigation and more depth on the harness and middleware. The AutoGen section's Microsoft pages are good and verified. Its AG2 Classic pages carry 7 blockers (a stub, broken examples, removed APIs, a false swarm claim). They also sit under a "Microsoft AutoGen" label, which undermines the otherwise careful separation.

#### Code verification results

How the code was run:
- **AG2 1.x:** `ag2` 1.2.1 venv. Every provider config's `create()` was patched to return `ag2.testing.TestClient` with scripted turns, and `input` was patched. Each block was executed and then driven (tools called, schemas parsed, Network graph run to `EV_CHANNEL_CLOSED`).
- **Microsoft AutoGen:** 0.7.5 venv. `OpenAIChatCompletionClient(...)` was constructed for real (which checks its kwargs) and then swapped for `ReplayChatCompletionClient` with function-calling `CreateResult`s. Teams, Swarm handoffs, GraphFlow loops, TeamTool/AgentTool, save/load state, the Core runtime and OTel spans were all exercised.
- **AG2 Classic:** `autogen` 0.14.1 venv. `ConversableAgent.generate_oai_reply` and human input were patched; OAI_CONFIG_LIST fixtures were provided; fragments that rely on earlier blocks were re-run with a context prelude.
- **C#:** .NET SDK 10.0.402 with NuGet `AutoGen` 0.2.3.
- **Python syntax:** all 129 Python blocks parse.

| Page | Code base | Blocks | Pass | Fail | Not run | Notes |
|---|---|---|---|---|---|---|
| ag2-guide/index.mdx | AG2 1.x | 1 | 1 | 0 | 0 | |
| ag2-guide/ag2_comprehensive_guide.md | AG2 1.x (+1 Classic) | 11 | 11 | 0 | 0 | Testing block needs `pytest-asyncio` (not mentioned) |
| ag2-guide/ag2_recipes.md | AG2 1.x | 6 | 6 | 0 | 0 | MCP recipe works; see sandbox note below |
| ag2-guide/ag2_production_guide.md, ag2_diagrams.md | — | 0 | — | — | — | Only text/dockerfile/mermaid blocks |
| autogen-guide/python/index.mdx | Microsoft | 1 | 1 | 0 | 0 | |
| autogen-guide/python/autogen_agentchat_guide.md | Microsoft | 10 | 10 | 0 | 0 | MCP needs `mcp<2` (confirmed: mcp 2.x raises ImportError) |
| autogen-guide/python/autogen_advanced_multi_agent.md | Microsoft | 4 | 4 | 0 | 0 | |
| autogen-guide/python/autogen_observability_python.md | Microsoft | 3 | 3 | 0 | 0 | Spans emitted |
| autogen-guide/python/autogen_comprehensive_guide.md | Classic | 52 | 45 | 5 | 2 | Fails: L890 `.process()`, L1058 bad kwarg, Docker default at L821/L847/L1298. Not run: Docker executor (needs `autogen[docker]` and a daemon) |
| autogen-guide/python/autogen_recipes.md | Classic | 10 | 5 | 5 | 0 | Fails: L888 `.process()`; Docker default at L83/L653/L778/L953 |
| autogen-guide/python/autogen_production_guide.md | Classic | 28 | 23 | 5 | 0 | Fails: A2A server (L565-580), A2A client (L605), LangChain (L904), LlamaIndex (L934), Docker default (L275). Many "passes" do nothing meaningful |
| autogen-guide/python/autogen_to_ag2_migration_guide.md | Classic + AG2 | 3 | 3 | 0 | 0 | |
| autogen-guide/dotnet.md | AutoGen.Net 0.2.3 | 1 (C#) | 0 | 1 | 0 | CS1061 `RegisterPrintMessage`: missing `using AutoGen.Core;` (upstream README has the same omission) |
| **Total** | | **130** | **112** | **16** | **2** | AG2 1.x 17/17 · Microsoft 18/18 · Classic 75/92 · C# 0/1 |

Sandbox note: both MCP stdio clients timed out launching `npx @modelcontextprotocol/server-filesystem` until `env=dict(os.environ)` was passed. The MCP SDK gives the subprocess a stripped environment, so behind a proxy `npx` cannot reach the registry. The code is correct; it is worth a one-line tip on both MCP sections.

Site checks:
- `node scripts/check-versions.mjs`: no ag2/autogen findings.
- `node scripts/check-links.mjs`: 0 broken (whole site).
- No page body starts with an H1.
- No class deep-dive or "Vol. N" pages remain, but there is one "Deep Dive" heading.

#### Issues, ranked

### Blocker

All seven are in the AG2 Classic pages of `autogen-guide/`.

| # | File:line | Problem | Fix |
|---|---|---|---|
| B1 | autogen-guide/python/autogen_comprehensive_guide.md:1250-1256 | Placeholder stub left in a published page: "[Content continues with extensive tool integration examples...] (Continuing in next part due to length...)". The table of contents (line 30) links to it. | Delete the section, or fill it with a short `register_function` / `@agent.register_for_llm` example. |
| B2 | autogen_comprehensive_guide.md:1053-1059 | `GroupChat(..., speaker_selection_method="manual", select_speaker_message_function=...)` raises `TypeError` (no such kwarg). `"manual"` means "ask a human". | Pass the callable directly: `speaker_selection_method=custom_speaker_selector`, with signature `(last_speaker, groupchat)`. |
| B3 | autogen_comprehensive_guide.md:885-891; autogen_recipes.md:883-889 | `initiate_chat(...)` returns `ChatResult`, so `.process()` raises `AttributeError`. | Drop `.process()` (it only belongs on `run()`'s `RunResponse`), or switch to `agent.run(...)`. |
| B4 | autogen_production_guide.md:565-580, 595-608 | `A2aAgentServer(agent_card={...dict...})` fails (`'dict' object has no attribute 'url'`; it expects `CardSettings`), and `server.start()` does not exist (methods are `build`, `build_starlette_app`). `A2aRemoteAgent` with sync `initiate_chat` raises `NotImplementedError`. | Use `app = A2aAgentServer(agent, url=...).build()` under uvicorn, and `await local.a_initiate_chat(remote_agent, ...)`. |
| B5 | autogen_production_guide.md:898-958 | "LangChain / LlamaIndex integration" imports removed APIs (`langchain.llms.OpenAI`, `llama_index.GPTVectorStoreIndex`) and the retired `text-davinci-003`, and integrates nothing: the LangChain LLM is never used. | Delete the section, or replace it with a real tool wrapper (`autogen.interop.Interoperability` for LangChain tools) on current packages. |
| B6 | autogen_production_guide.md:516-520 | `requirements.txt` pins `ag2[openai]==0.2.x`: an invalid specifier for an ancient release, contradicting the page's own `autogen[openai]` install. This is exactly the AG2/Classic mix-up the page warns about. | `autogen[openai]>=%%version=autogen%%,<0.15` (as in the migration guide line 34). |
| B7 | autogen_comprehensive_guide.md:1553, 1586; autogen_to_ag2_migration_guide.md:58 | Claims `SwarmAgent` / `initiate_swarm_chat` live in `autogen.agentchat.contrib.swarm_agent`. That module existed in ag2 0.13.0 but is absent from ag2 0.14.0 and `autogen` 0.14.1. | Say swarms were removed in 0.14.0. Use group-chat patterns (`autogen.agentchat.group.patterns`) with handoffs, or pin `ag2==0.13.*`. Keep "moved in 0.13.0" only as history (ag2-guide/index.mdx:67 is fine). |

### Major

| # | File:line | Problem | Fix |
|---|---|---|---|
| M1 | astro.config.mjs:117; autogen-guide/python/index.mdx:2, 28 | **Separation.** The sidebar entry "Microsoft AutoGen (Python)" leads to a section whose comprehensive guide, recipes, production and diagrams pages are AG2 Classic. Slugs `autogen_comprehensive_guide` etc. reinforce the confusion. | Move the Classic pages to `/ag2-guide/classic/…` (add redirects), or relabel the entry "AutoGen (Python): Microsoft 0.7 and AG2 Classic" and split the sidebar into two groups. |
| M2 | src/data/frameworks.ts:88-98; guides.md:240-251; frameworks.md:248-274 | **Separation.** There is one home card for both projects, with the tagline "GroupChat and A2A" and a link only to `/autogen-guide/`. guides.md never links the AG2 1.x guide and has a duplicate link (243/244). frameworks.md sells AG2 on Classic features ("GroupChat", "conversation-based") but links to the 1.x guide. | Add a separate AG2 card linking `/ag2-guide/` and retitle the AutoGen card "Microsoft AutoGen (archived)". Add AG2 1.x links in guides.md. Update frameworks.md to 1.x terms (Agent, Network, channels). |
| M3 | ag2-guide/index.mdx:2 | Title "AG2 (formerly AutoGen)" implies a rename, the narrative the rewrite removed elsewhere. | Title "AG2"; put "community fork of AutoGen 0.2" in the description. |
| M4 | ag2_comprehensive_guide.md:349-394; autogen_comprehensive_guide.md:1570-1590; autogen_to_ag2_migration_guide.md (paths A/B) | Migration and Classic-status content appears three times, with slightly different wording. | Keep the migration guide as the single source; replace the other two with a link. |
| M5 | ag2_comprehensive_guide.md:392; ag2_production_guide.md:61; ag2_comprehensive_guide.md:127 | `CodeExecutionTool` and `ShellTool` are **provider-executed server-side** tools (`ag2.tools.builtin.*`; ShellTool runs in OpenAI's container), not replacements for Classic local executors. The local equivalents are `SandboxCodeTool` / `SandboxShellTool` with `LocalEnvironment` / `DockerEnvironment` / `DaytonaEnvironment`. The production advice "prefer SandboxCodeTool over LocalEnvironment and ShellTool" mixes the two. | Step 7: "Code executors → `SandboxCodeTool` / `SandboxShellTool` with a `DockerEnvironment` (or `LocalEnvironment` for trusted code)". Security: "prefer Docker/Daytona environments over `LocalEnvironment`; `CodeExecutionTool`/`ShellTool` run on the provider." |
| M6 | ag2_comprehensive_guide.md:57, 76, 119, 146, 181, 236, 278, 366; ag2_recipes.md:21, 55, 82, 106, 134, 175; ag2-guide/index.mdx:80; autogen_agentchat_guide.md:47; autogen-guide/python/index.mdx:49; autogen_to_ag2_migration_guide.md:43, 70, 101; autogen_comprehensive_guide.md:1523, 1538; autogen_production_guide.md:672 | Stale OpenAI model `gpt-4.1-mini`. The OpenAI SDK lists gpt-5.x and gpt-6 families; the rest of the site uses `gpt-5-mini` / `gpt-5.4-mini`. | Replace with `gpt-5.4-mini` (or the site-standard `gpt-5-mini`). |
| M7 | ag2_comprehensive_guide.md:59, 308 | Stale `claude-sonnet-4-6`. | `claude-sonnet-5-5`. |
| M8 | autogen_comprehensive_guide.md:110, 136, 140, 154, 173, 198, 210, 222, 552, 571; autogen_production_guide.md:533, 731, 915 | Retired or very old models in Classic code: `gpt-4`, `gpt-3.5-turbo`, `claude-3-5-sonnet-20240620`, `gemini-pro`, `llama3`, `text-davinci-003`. | `gpt-5.4-mini`, `claude-sonnet-5-5`, `gemini-2.5-flash` / `gemini-3.5-flash`, `llama3.1`. Drop `text-davinci-003`. |
| M9 | autogen_comprehensive_guide.md:821, 847, 1298; autogen_recipes.md:83, 653, 778, 953; autogen_production_guide.md:275 | `UserProxyAgent` without `code_execution_config` defaults to Docker execution and raises `RuntimeError` when Docker is not running (8 blocks). | Add `code_execution_config=False` (or use `ConversableAgent`) wherever no code runs. |
| M10 | autogen_production_guide.md:120-152, 682-710, 860-894, 964-1022 | Filler. `CostTrackingAgent.track_usage` is never called; `CachingAgent` caches a hard-coded string; the async "examples" only `asyncio.sleep`; the health check counts dict keys. | Replace with real Classic features: `autogen.runtime_logging.start()/stop()`, `autogen.opentelemetry` (tracing), `gather_usage_summary` / `ChatResult.cost`, `a_initiate_chat` for concurrency, `cache_seed` / `Cache.disk()`. |
| M11 | autogen_comprehensive_guide.md:439-495, 1451-1508 | Padding: role-prompt and few-shot "examples" with no API content (strings or bare agent constructors). | Cut to one example, or move to recipes. |
| M12 | ag2-guide (coverage) | One line or nothing on the agent harness (`KnowledgeConfig`, assembly, compaction; production:50), middleware/observers (production:69), DI (`Depends`, `Inject`, `Variable`, `dependencies=` / `variables=`), multimodal inputs (`ImageInput`, `DocumentInput`, …), `Task`, `Plugin`, `ag2.eval`. | Add short verified sections for the harness, middleware/observers and DI, plus one multimodal example. |
| M13 | autogen-guide Microsoft pages (coverage) | Missing: memory (`autogen_core.memory.ListMemory`, `autogen_ext.memory.*`), model context (`BufferedChatCompletionContext`), a `CodeExecutorAgent` example, Magentic-One usage, AutoGen Studio (mentioned only at agentchat_guide:268). | Add short sections; since the project is archived, keep them brief and maintenance-oriented. |
| M14 | astro.config.mjs:113-121 | The sidebar lists only index pages; none of the 4 AG2 subpages or 8 AutoGen Python subpages appear (reachable only via cards). Other guides (e.g. LangGraph) list their subpages. | Add subpage entries under "AG2" and the two AutoGen groups. |

### Minor

| # | File:line | Problem | Fix |
|---|---|---|---|
| m1 | ag2_comprehensive_guide.md:339 | `@pytest.mark.asyncio` needs `pytest-asyncio`, which is not mentioned. | Add `pip install pytest pytest-asyncio`, or use `@pytest.mark.anyio`. |
| m2 | ag2_production_guide.md:57 | `A2AReconnectError` is not exported from `ag2.a2a`. | Name the path: `ag2.a2a.errors.A2AReconnectError`. |
| m3 | ag2_comprehensive_guide.md:60 | `gemini-2.5-flash`; newer Gemini models exist (`gemini-3.5-flash` is used elsewhere on the site). | Update after checking the live Gemini list. |
| m4 | ag2_comprehensive_guide.md:1-5, ag2_recipes.md:1-5, ag2_production_guide.md:1-5, ag2_diagrams.md:1-5; autogen-guide/index.mdx:1-8 | Frontmatter lacks `language: python`. | Add it. |
| m5 | ag2-guide/index.mdx:30; ag2_comprehensive_guide.md:351 | The upstream ag2 README says `pip install ag2-classic`, but no such PyPI package exists (the ag2-classic README says `autogen`). Readers will meet the conflict. | Add one line: "`ag2-classic` is not on PyPI; install `autogen`." |
| m6 | ag2_comprehensive_guide.md:270-288; autogen_agentchat_guide.md:183-198 | MCP stdio subprocesses get a stripped environment (`npx` fails behind proxies). | Tip: pass `env=` on `MCPStdioServerConfig` / `StdioServerParams`. |
| m7 | ag2_production_guide.md:244-249 | Observability is bullets only (no OTel setup, no `ag2.usage` example). | Add one verified tracing snippet and one usage snippet. |
| m8 | autogen_agentchat_guide.md:259-266; autogen_advanced_multi_agent.md:127 | Model clients and executors need extras that are not stated (`autogen-ext[anthropic]`, `[ollama]`, `[azure]`, `[docker]`, `[grpc]`). | Add an extras column. |
| m9 | autogen_comprehensive_guide.md:656, 680 | `DockerCommandLineCodeExecutor` needs `autogen[docker]`, which is not stated. | Add the extra. |
| m10 | autogen-guide/dotnet.md:30-31 | The snippet does not compile without `using AutoGen.Core;` (verified with AutoGen 0.2.3). | Add the using. |
| m11 | autogen_comprehensive_guide.md:99, 714 | "In modern AutoGen…" on a Classic page. | Say "In AG2 Classic…". |
| m12 | autogen_comprehensive_guide.md:130, 150 | "By default, AutoGen looks for…" / "automatically finds and loads this file". `config_list_from_json` needs the file name or env var. | Remove the auto-discovery claim. |
| m13 | autogen_comprehensive_guide.md:497 | Heading "Agent Configuration Deep Dive". | "Agent configuration". |
| m14 | autogen_diagrams.md:31, 731; autogen_production_guide.md:455, 1026 | Unqualified "AG2" on Classic pages. | "AG2 Classic". |
| m15 | autogen_recipes.md:2-3, 10, 1106; autogen_production_guide.md:3, 10; autogen_diagrams.md:3, 9, 712 | Title-case marketing descriptions ("Enterprise-Grade…", "Real-world Code Examples…"), exclamation marks. | Plain sentence-case descriptions; drop the exclamations. |
| m16 | autogen_recipes.md:1102 | Refers to `max_rounds`; GroupChat uses `max_round`, and these recipes use `max_turns`. | Fix the name. |
| m17 | autogen_comprehensive_guide.md:349, 528, 824 | `is_termination_msg=lambda x: x.get("content", "")…` breaks when content is `None` (tool-call messages). | `(x.get("content") or "")`. |
| m18 | autogen_comprehensive_guide.md:110-111, 135-141, 185, 198, 210 | Hard-coded API keys in examples. | Read from the environment. |
| m19 | autogen_comprehensive_guide.md:307, 917; autogen_production_guide.md:435 | American spellings: optimized, specialized, behavior. | optimised, specialised, behaviour. |
| m20 | autogen-guide/index.mdx:24, 34 | "October 2026 archived": the archived status is confirmed by the upstream README, but the date could not be independently verified here. | Cite the README or the announcement. |

#### Top 5 improvements

1. **Fix the separation in the information architecture** (M1-M3, M14):
   - Move the AG2 Classic pages under `/ag2-guide/classic/` with redirects, so `autogen-guide/` is purely Microsoft AutoGen.
   - Give AG2 its own home card and sidebar group with subpages.
   - Retitle the AG2 index "AG2".
   - Fix `guides.md` and `frameworks.md`.
2. **Clear the 7 Classic blockers and the Docker-default failures** (B1-B7, M9): remove the stub, fix the GroupChat kwarg, `.process()`, the A2A server/client code and the `ag2==0.2.x` pin; delete the LangChain/LlamaIndex section; correct the swarm claim; add `code_execution_config=False`. That takes the Classic pass rate from 75/92 to about 90/92.
3. **Rewrite the Classic production guide around real Classic features** (M10, M11): `runtime_logging`, `autogen.opentelemetry`, `gather_usage_summary`, disk cache, Docker executors and the correct A2A server API. Cut the filler and the prompt-only padding, and keep one migration text (M4).
4. **Update model IDs across both guides** (M6-M8, m3): `gpt-5.4-mini` (or the site-standard `gpt-5-mini`), `claude-sonnet-5-5`, current Gemini. Also correct the AG2 code-execution guidance (M5), the only substantive accuracy problem in the 1.x guide.
5. **Deepen AG2 1.x coverage and production** (M12, m7): verified sections on the harness (knowledge, assembly, compaction), middleware/observers, dependency injection and multimodal inputs, plus a production page with an OTel tracing snippet and an A2A/MCP serving example under uvicorn.

---

### Assessment: Microsoft Agent Framework guide

Scope: `src/content/docs/microsoft-agent-framework-guide/`. That is 45 pages (31k lines): 9 root pages, 23 Python, 9 .NET and 4 Go. Assessed on 2026-10-21 against the versions in `src/data/versions.json`: `agent-framework` 1.22.0, `Microsoft.Agents.AI` 1.25.0 and `github.com/microsoft/agent-framework-go` v0.2.0. All paths below are relative to the guide folder unless they start with `src/` or `/`.

Scratch work is in `/tmp/claude-0/-home-user-AgentGuides/55b0d51f-358a-5617-8094-250bd4df3d90/scratchpad/assess-maf/`:

- `pyall.py`: static check of Python blocks (syntax, imports, keyword arguments).
- `sigchk.py`: checks signature-listing blocks against `inspect.signature`.
- `run.txt`, `runcomp.txt`, `split.txt`: every Python block executed against a stub chat client.
- `cs.txt`: C# compile results.
- `go/`: extracted Go programs, plus `go.txt`.
- `anchors.py`: in-page anchor check.

#### Scorecard

| # | Criterion | Score | Justification |
|---|---|---|---|
| 1 | Correctness of code | 3/5 | **C#:** all 67 blocks compile. **Go:** all 21 programs pass `go vet`, and the Go testing-page tests pass. **Python:** 20 runnable blocks break against 1.22.0. These include the first Human-in-the-loop (HITL) example, the main streaming example, both switch-case and multi-selection orchestration examples, and the A2A client and auth examples. |
| 2 | Currency | 2/5 | The version placeholders are clean. However, about 75 hard-coded `gpt-4o` / `gpt-4o-mini` / `gpt-4-turbo` IDs remain, and `claude-sonnet-4-5` appears 4 times. The Python comprehensive guide opens with "What's new in 1.14.0", and its class reference says it was "checked against 1.14.0". The legacy `agent-framework-azure-ai` package (stuck at rc6) is recommended in 3 places. |
| 3 | Accuracy of prose | 3/5 | The class reference is mostly precise. But there are clear false claims: A2A hosting is called "out of scope" (it is `A2AExecutor`), and A2A is said to have "exactly one class". `stream.send_response()` is presented as real on two pages and called non-existent on two others. `FileHistoryProvider` is called "stable since 1.5.0" (it is experimental). The Go first-release date is wrong, and the GA dates disagree between pages. |
| 4 | Coverage | 3/5 | **Python:** very thorough (harness, FIDES, skills, compaction, evaluation, durable task, DevUI). **.NET:** thin. Nothing on the stable `Microsoft.Agents.AI.Harness`, `.Anthropic` or `.GitHub.Copilot` packages, nor on DevUI or evaluation; the RBAC page is a 19-line stub. **Go:** proportionate and honest about its gaps. |
| 5 | Structure and navigation | 2/5 | 64 relative links on non-index pages resolve to 404 in the built site. The sidebar lists 6 of 23 Python pages and none of the root, .NET or Go sub-pages. The quick-start page is orphaned. `migration_notice.md` is obsolete repository-era text. The 3,718-line Python comprehensive guide duplicates topic pages and carries 20 errata or "Correction" notes. |
| 6 | Consistency | 3/5 | Frontmatter is complete everywhere. Experimental/stable labels contradict each other, page titles use four different formats, Azure role names differ between pages, and there are about 40 American spellings in prose. |
| 7 | Production readiness | 3/5 | **Python:** the production page's §10 (operational patterns specific to `agent_framework`) is real, framework-specific advice. **.NET:** §§5–8 are generic bullet lists. No production page covers Foundry hosted agents, Azure Functions / durable agents, prompt-injection defence (FIDES) or running evaluation in CI. |

**Overall verdict.** The C# and Go material is in good shape and compiles cleanly. The Python material is broad and mostly verified, but about 20 examples still fail against 1.22.0, including core HITL, streaming and routing snippets. Several status claims are false, and broken relative links plus a sidebar that hides three-quarters of the pages make the guide hard to navigate. Fixing the 16 blockers and restructuring the Python comprehensive guide would lift it to a solid 4.

#### Code verification

**Method.**

- **Python:**
  - Every ```` ```python ```` block was parsed with top-level await allowed.
  - `agent_framework*`, `a2a` and `mcp` imports were resolved.
  - Keyword arguments were checked against `inspect.signature`.
  - Every block was then executed with `OpenAIChatClient` / `FoundryChatClient` replaced by a stub echo client (`scratchpad/maf/stub.py`), using the venv with `agent-framework` 1.22.0 and all sub-packages.
  - Failures caused only by missing context were not counted as code faults: names defined in earlier blocks, missing API keys, Azure credentials, MCP servers, YAML files, `tiktoken`, or the harness's own `asyncio.run` nesting.
  - Signature-listing blocks (`Class(param: type = …)` pseudo-code tagged `python`) were checked separately against the real signatures.
- **C#:** `cscheck.py` with .NET 10, against Microsoft.Agents.AI 1.25.0 and its companion packages. Blocks that reference types from earlier blocks on the same page were re-checked with `--shared`.
- **Go:** every block with a `package` clause was extracted into module `xcheck` (v0.2.0) and run through `go vet` and `go build` with Go 1.26.0. The testing page's three blocks were combined into one package and run with `go test`.

**Totals.**

| Language | Blocks | Checked as code | Pass | Fail | Notes |
|---|---|---|---|---|---|
| Python | 744 | 671 (+73 signature listings) | 651 | 20 | 1 syntax error, 3 import/keyword failures, 16 runtime API errors. 321 blocks ran clean end to end on the stub. Signature listings: no wrong parameter names, a few new parameters omitted. |
| C# | 67 | 67 | 67 | 0 | Two pages need page-level compilation because types are shared across blocks. |
| Go | 26 | 21 programs | 21 | 0 | 5 fragments not compiled. `go test` passes against the fake provider. |

**Per page.** Python "pass" means no genuine error was found; for fragments, that is a static check only.

| Page | Py blocks | Py fail | C# | Go |
|---|---|---|---|---|
| index.mdx | – | – | – | – |
| microsoft_agent_framework_a2a_protocol.md | 4 (1 sig) | **2** | – | – |
| microsoft_agent_framework_comprehensive_guide.md | 13 | 0 | 7/7 | – |
| microsoft_agent_framework_enterprise_2025.md | 22 | **1** | 3/3 | – |
| microsoft_agent_framework_graphs_declarative.md | 33 (1 sig) | **1** | – | – |
| microsoft_agent_framework_production_guide.md | 12 | 0 | – | – |
| microsoft_agent_framework_recipes.md | 9 | 0 | 2/2 | – |
| microsoft_agent_framework_diagrams.md, migration_notice.md | 0 | – | – | – |
| python/index.mdx | 1 | 0 (runs on stub) | – | – |
| python/…_installation_and_quickstart.md | 5 | 0 (4 run on stub) | – | – |
| python/…_comprehensive_guide.md | 154 (53 sig) | **6** | – | – |
| python/…_2025_features.md | 15 | **1** | – | – |
| python/…_advanced.md | 32 | **3** | – | – |
| python/…_checkpointing.md | 21 | 0 | – | – |
| python/…_compaction.md | 46 (2 sig) | **1** | – | – |
| python/…_evaluation.md | 27 | **1** | – | – |
| python/…_functional_workflows.md | 19 (2 sig) | 0 | – | – |
| python/…_harness.md | 36 (8 sig) | 0 | – | – |
| python/…_hitl.md | 14 | **1** (2 sites) | – | – |
| python/…_mcp.md, _middleware.md, _observability.md, _sessions.md, _skills.md, _tools.md, _recipes.md, _production_guide.md, _diagrams.md, _sdk_migration_notice.md | 25 / 25 / 25 / 21 / 26 / 34 / 28 / 15 / 4 / 2 | 0 | – | – |
| python/…_model_providers.md | 30 | **1** | – | – |
| python/…_orchestration.md | 44 | **2** | – | – |
| dotnet/index.mdx | – | – | 1/1 | – |
| dotnet/…_comprehensive_guide.md | – | – | 19/19 | – |
| dotnet/…_recipes.md | – | – | 12/12 | – |
| dotnet/…_2025_features.md | – | – | 16/16 | – |
| dotnet/…_production_guide.md | – | – | 5/5 | – |
| dotnet/…_advanced.md | – | – | 1/1 | – |
| dotnet/microsoft_agent_streaming_server_dotnet.md | – | – | 1/1 | – |
| go/index.mdx | – | – | – | 1/1 |
| go/…_comprehensive_guide.md | – | – | – | 15/15 (1 fragment) |
| go/…_workflows.md | – | – | – | 4/4 (2 fragments) |
| go/…_testing.md | – | – | – | 3/3 + `go test` ok (1 fragment) |

**Repository checks.**

- `node scripts/check-versions.mjs | grep microsoft-agent`: clean.
- `node scripts/check-links.mjs`: 0 broken. However, the checker resolves `./x/` against the source folder rather than the page URL, so it misses the relative-link breakage described in B1.
- In-page anchors (123 checked): all resolve.

#### Issues

### Blocker

**B1. 64 relative links on non-index pages 404 in the browser.**

- Problem: a link such as `](./microsoft_agent_framework_python_evaluation/)` on `/python/microsoft_agent_framework_python_recipes/` resolves to `/python/microsoft_agent_framework_python_recipes/microsoft_agent_framework_python_evaluation/`. I confirmed this in `dist/…/python_recipes/index.html`, which still contains `href="./microsoft_agent_framework_python_evaluation/"`.
- Locations:
  - python/…_comprehensive_guide.md: 471, 511, 769, 891, 922, 1421, 1461, 1672, 1719, 1775, 1814, 2568, 3567, 3709–3718
  - python/…_recipes.md: 400, 460, 511, 754, 1030, 1231, 1525 (×9)
  - python/…_tools.md: 10 (×2), 215, 232, 596
  - python/…_functional_workflows.md: 605–608
  - python/…_sessions.md: 720–722
  - python/…_mcp.md: 213, 248, 506
  - python/…_orchestration.md: 80, 245, 295
  - python/…_advanced.md: 395, 1112
  - python/…_model_providers.md: 366, 700
  - python/…_observability.md: 234
  - python/…_compaction.md: 778
  - python/…_evaluation.md: 950
  - python/…_checkpointing.md: 619
  - microsoft_agent_framework_graphs_declarative.md: 883
- Fix: rewrite them as root-absolute `/microsoft-agent-framework-guide/python/…/` links, as AGENTS.md requires, and teach `check-links.mjs` to flag `./` links in non-index pages.

**B2. `stream.send_response()` is presented as real, but it does not exist.**

- Locations: python/…_recipes.md:88 and :938, python/…_comprehensive_guide.md:583.
- Problem: `ResponseStream` has no such method in 1.22.0. hitl.md:349 and skills.md:622 correctly say so, so the pages contradict each other.
- Fix: replace the text with the `response.user_input_requests` → reply-on-session loop from hitl.md.

**B3. Streaming example calls two APIs that do not exist.**

- Location: python/…_comprehensive_guide.md:576–577.
- Problem: `await stream.get_response()` raises `AttributeError`; the method is `get_final_response()`. `final.finish_reasons` is not an attribute of `AgentResponse`.
- Fix: use `get_final_response()` and print `final.text` or `final.usage_details`.

**B4. The first HITL example fails at construction.**

- Locations: python/…_hitl.md:61 and :371.
- Problem: `Researcher()` raises `TypeError: Executor.__init__() missing 1 required positional argument: 'id'`.
- Fix: use `Researcher(id="researcher")`.

**B5. Switch-case and multi-selection orchestration examples fail at `build()`.**

- Locations: python/…_orchestration.md:512–562 and :570–610.
- Problem: building raises `TypeCompatibilityError`, because a `SupportTicket` (or `AnalysisTask`) dataclass is routed straight to `Agent` targets, which accept only `str`, `Message` or a list of these. The same pattern appears at :956.
- Fix: route to an executor that converts the dataclass to text, or send `ticket.text` with a condition on a wrapper.

**B6. `WorkflowContext[NoReturn, str]` is rejected.**

- Locations: python/…_comprehensive_guide.md:1068, 1109, 1114, 1171, 1176 (imported at 1059, 1088, 1150).
- Problem: raises `ValueError: OutT contains invalid type entry: typing.NoReturn`.
- Fix: use `typing_extensions.Never` (verified to work) or `WorkflowContext[None, str]`.

**B7. Duplicate executor IDs.**

- Locations: python/…_comprehensive_guide.md:1349 and :1352 (`WorkflowExecutor(inner, id="draft-pipeline")` constructed twice); :2613 and :2614 (`Normalize(id="norm")` and `Count(id="count")` constructed twice).
- Problem: both raise `ValueError: Duplicate executor ID`. In the second example, `Normalize` also sends `str` to a `Count` that expects `list[str]`.
- Fix: create each executor once, assign it to a variable and reuse it; make the types agree.

**B8. The A2A page's code and its central claim are both wrong.**

- Locations and problems:
  - microsoft_agent_framework_a2a_protocol.md:76: `AgentCard.from_url` does not exist in `a2a-sdk` 1.2.3. Fix: use `A2ACardResolver`.
  - :87–95: `BearerTokenCredentials` is not exported, and `AuthInterceptor` takes `credential_service=`, not `credentials=`. Fix: use a `CredentialService` (e.g. with `InMemoryContextCredentialStore`).
  - :99–101: "Serving a framework Agent as an A2A endpoint is out of scope" is false. `agent_framework.a2a.A2AExecutor(agent, stream=False, run_kwargs=None)` exists and its docstring shows a full server example.
- Fix: correct both examples and add an "Exposing an agent with `A2AExecutor`" section.

**B9. `ClaudeAgent(options=...)` raises `TypeError`.**

- Location: python/…_model_providers.md:184–186.
- Problem: `RawClaudeAgent.__init__` has no `options` parameter; the keyword is `default_options=`. The model ID on that line is also stale.
- Fix: `ClaudeAgent(default_options=ClaudeAgentOptions(model="sonnet", permission_mode="default"))`.

**B10. The evaluation example asserts the wrong value.**

- Location: python/…_evaluation.md:240.
- Problem: with the default split, `EvalItem(conversation=…).query` returns `"What's 2+2? Square that."` (all user turns), not `"Square that."`, so the `AssertionError` fires. The prose at :236 describes the wrong behaviour.
- Fix: correct the assertion and the explanation.

**B11. Three broken examples in the advanced page.**

- python/…_advanced.md:241: `FinishReason.STOP` raises `AttributeError`, because `FinishReason` is a `NewType`. Fix: use `finish_reason="stop"`.
- :954: `Content.from_text_reasoning("…")` raises `TypeError`, because the method is keyword-only. Fix: `Content.from_text_reasoning(text="…")`.
- :413, :434, :493, :513: `agent.run(..., tenant_id="acme")` raises `TypeError`, and `SessionContext` has no `.kwargs`. Fix: pass tenant data through `session.state` (or `function_invocation_kwargs` for tools) and read it in `before_run`.

**B12. `FunctionExecutor(lambda …)` raises `ValueError: must have a type annotation`.**

- Locations: microsoft_agent_framework_graphs_declarative.md:593–596 and python/…_2025_features.md:57–58.
- Fix: use annotated `def` functions or `@executor`, or pass `input=` and `output=` types.

**B13. `from azure.identity.aio import DeviceCodeCredential` raises `ImportError`.**

- Location: microsoft_agent_framework_enterprise_2025.md:561.
- Problem: the async package has no device-code (or interactive browser) credential.
- Fix: use the sync `azure.identity.DeviceCodeCredential`, or `AzureCliCredential` from `.aio`.

**B14. Syntax error.**

- Location: python/…_compaction.md:1037–1042.
- Problem: an invalid list comprehension (two elements before `for`), followed by a comment that apologises for it.
- Fix: delete the dead `messages = [...]` block and keep the `flat` loop.

**B15. False status claim for `FileHistoryProvider`.**

- Location: python/…_sessions.md:16.
- Problem: says `FileHistoryProvider` has been "stable since 1.5.0". In 1.22.0 it carries `ExperimentalFeature.FILE_HISTORY`, as compaction.md:573 and advanced.md:679 correctly say.
- Fix: mark it experimental.

**B16. Wrong release date for the Go SDK.**

- Location: go/index.mdx:16.
- Problem: says "first published 20 October 2026". The Go proxy shows v0.1.0 on 2026-09-01, v0.1.1 on 2026-10-13 and v0.2.0 on 2026-10-20.
- Fix: "first published 1 September 2026; v0.2.0 on 20 October 2026".

### Major

**M1. Stale model IDs.**

- OpenAI `gpt-4o` / `gpt-4o-mini` (about 73 occurrences in 26 files). Worst offenders:
  - python/…_compaction.md: 331, 345, 404, 430, 469, 479, 772, 796, 804, 832
  - python/…_comprehensive_guide.md: 132, 326, 334, 362, 372, 2787, 2817, 2835, 3678
  - microsoft_agent_framework_enterprise_2025.md: 154, 362, 494, 553, 596, 711, 801, 840
  - microsoft_agent_framework_comprehensive_guide.md: 117, 142, 145, 154, 227, 302
  - microsoft_agent_framework_production_guide.md (5 occurrences)
  - dotnet/index.mdx:107
  - dotnet/…_comprehensive_guide.md: 87, 111, 704
  - dotnet/…_recipes.md:40
  - dotnet/microsoft_agent_streaming_server_dotnet.md:35
  - python/…_installation_and_quickstart.md:86
  - python/…_orchestration.md: 228, 604, 605
  - go/…_comprehensive_guide.md:500
- Other stale IDs:
  - `gpt-4-turbo`: microsoft_agent_framework_diagrams.md:602
  - `o4-mini`: microsoft_agent_framework_enterprise_2025.md:802
  - `claude-sonnet-4-5`: python/…_model_providers.md:174, 185, 287 and go/…_comprehensive_guide.md:573
  - `gemini-2.5-flash`: go/…_comprehensive_guide.md:609
  - `llama3.1`: python/…_model_providers.md:199, 207, 289, 704
- Fix: use the IDs the rest of the site uses (`gpt-5.4-mini`, already used on the Go pages; `claude-sonnet-5-5`; `gemini-3.5-flash`) or neutral deployment names. I could not reach live provider model lists without credentials.

**M2. Navigation gaps in the sidebar and the quick-start page.**

- Problem: `astro.config.mjs:226–239` shows only 6 of 23 Python pages and no root, .NET or Go sub-pages. `python/…_installation_and_quickstart.md` has zero inbound links: it is not in the sidebar and no page links to it.
- Fix: autogenerate or list every page per language, and link the quick-start from python/index.mdx "Install, minimal example".

**M3. `migration_notice.md` is obsolete.**

- Problem: the whole page is repository-era content: `cd python`, `GUIDE_INDEX.md`, `mv … _ARCHIVED`, "No longer actively maintained" and emoji headings. It is still linked as the Semantic Kernel migration reference from microsoft_agent_framework_comprehensive_guide.md:101 and dotnet/…_comprehensive_guide.md:776.
- Fix: delete it, add a redirect, and point the SK links to Microsoft's SK migration guide.

**M4. The Python comprehensive guide is a 3,718-line catch-all.**

- Problems:
  - "What's new in 1.14.0" sits at the top (:32).
  - Orchestration is covered twice on the page (:388 and :1595) and again on the orchestration page.
  - The `TodoProvider` and `AgentModeProvider` sections (:2160–2423) repeat harness.md.
  - The "Class & API Reference" (:2573–3564), "checked against 1.14.0", is the folded-in material from the deleted per-class pages.
  - An `azure-ai-agents` appendix (:3565) covers a separate package.
- Fix: drop the 1.14.0 section; replace the duplicated sections with links; move reference entries onto their topic pages or trim them to tables; move the `azure-ai-agents` appendix to the add-on page.

**M5. Functional workflows are documented in four places.**

- Locations: python/…_functional_workflows.md, python/…_advanced.md:751–942, microsoft_agent_framework_graphs_declarative.md:879, and the comprehensive guide's reference section.
- Fix: keep the dedicated page and link to it from the others.

**M6. Errata and "earlier drafts" meta-notes throughout.**

- Problem: 20 of these in python/…_comprehensive_guide.md alone, plus python/index.mdx:29–31 and :163, python/…_2025_features.md:8, a2a_protocol.md:7 and :103–111, and graphs_declarative.md:919–926. A new reader does not need a list of APIs that never existed.
- Fix: delete them and keep history in revision tables only.

**M7. The A2A "only one class" claim is repeated.**

- Locations: python/…_2025_features.md:41 ("exposes exactly one class") and python/index.mdx:129.
- Fix: list `A2AAgent`, `A2AAgentSession`, `A2AExecutor` and `A2AServiceSessionId`.

**M8. Contradictory experimental/stable labels.**

- python/…_comprehensive_guide.md:2160 and :2322: the headings say "(Experimental HARNESS)", but the next lines say "Stable since 1.12.0" (correct for `TodoProvider` and `AgentModeProvider`).
- python/…_recipes.md:1032 also says "(Experimental)".
- Fix: say stable for `TodoProvider` and `AgentModeProvider`, and experimental for `TodoFileStore`, `MemoryContextProvider`, `FileAccessProvider`, `BackgroundAgentsProvider` and `AgentLoopMiddleware` (verified via `__feature_id__`).

**M9. Legacy `agent-framework-azure-ai` package (last release 1.0.0rc6) is recommended.**

- Locations: python/…_installation_and_quickstart.md:197 (says to install it for `agent_framework.foundry`), python/…_sdk_migration_notice.md:107, python/…_comprehensive_guide.md:511.
- Fix: replace with `agent-framework-foundry`.

**M10. .NET coverage gaps.**

- Problem: nothing on the stable NuGet packages `Microsoft.Agents.AI.Harness` 1.25.0, `.Anthropic` 1.25.0 and `.GitHub.Copilot` 1.25.0, nor on `.DevUI`; no evaluation; skills and compaction get one or two mentions each.
- Location: dotnet/microsoft_agent_framework_azure_rbac.md is a 19-line stub, yet dotnet/index.mdx:57 advertises it as a "20 min" step on "Managed Identity, Entra ID". It also names "Azure AI Developer", where the enterprise page uses "Azure AI User".
- Fix: add Harness and provider sections, and expand the RBAC page or merge it into the enterprise page.

**M11. The .NET advanced page promises more than it contains.**

- Location: dotnet/…_advanced.md:13–16 lists "Agent orchestration and HITL", but the page delivers one Persistent Agents example and two bullets (:45–47).
- Problem: it also centres the classic `Azure.AI.Agents.Persistent` service rather than `Microsoft.Agents.AI.Foundry`.
- Fix: write the promised sections or cut the list; show Foundry first.

**M12. Production pages lack framework-specific guidance.**

- Problem: dotnet/…_production_guide.md §§5–8 (:288–354) are generic bullets. None of the three production pages covers Foundry hosted agents (`agent-framework-foundry-hosting` `ResponsesHostServer`), Azure Functions or durable agents, FIDES prompt-injection defence, or evaluation in CI.
- Fix: add these, reusing material from enterprise_2025.md and evaluation.md.

**M13. Inconsistent release history.**

- Problem: GA is given as "April 3-7, 2026" (index.mdx:118, dotnet/index.mdx) and "2026-04-16" (python/index.mdx:187). PyPI and NuGet both show 1.0.0 on 2026-04-02. "November 2025 | 1.0 Preview" is also wrong: the first PyPI preview, 1.0.0b251001, was released in October 2025. dotnet/index.mdx labels April 10 as ".NET 1.0.1", but NuGet shows 1.1.0 on 2026-04-10.
- Fix: correct the dates from the registries.

### Minor

**Wording, claims and stale labels**

- index.mdx:95 and dotnet/index.mdx "Stable APIs with LTS commitment": no LTS policy is cited. Link the source or drop "LTS".
- index.mdx:96: says Bedrock, Gemini and Ollama are "first-party service connectors" for .NET, but there are no `Microsoft.Agents.AI.*` packages for them on NuGet. Say they come through `IChatClient` implementations.
- python/index.mdx:3 description "Class & API Reference covering 200+ classes": an unverifiable count and leftover deep-dive framing. Remove it.
- dotnet/index.mdx:68 and :76: the "Feature tour (2025 preview)" labels are stale. Use "Feature tour".
- dotnet/…_diagrams.md:26 says `AgentThread`. Use `AgentSession`.
- microsoft_agent_framework_diagrams.md:47, :101 and :434 show `ChatAgent` and `@ai_function`, both non-existent. Use `Agent`/`ChatClientAgent` and `@tool`.

**Titles, file names and spelling**

- Page titles use four formats (" - ", " — ", "Python -", "(Python) —"). The titles and file names "2025 / 2026 Features", `enterprise_2025` and `_2025_features` carry dates. Normalise to "Microsoft Agent Framework (Python) — X" and drop the year.
- About 40 American spellings in prose (visualization, serialization, behavior, labeled, optimization). Concentrated in python/…_comprehensive_guide.md (17) and dotnet/…_production_guide.md (4). Convert them, leaving code identifiers alone.

**Small formatting and content fixes**

- migration_notice.md:7–8 and dotnet/…_advanced.md:47–48: a blank line is missing before a `##` heading. Add it (moot if migration_notice.md is deleted).
- python/…_functional_workflows.md:296: a code block inside a blockquote. Move it out of the quote.
- python/…_observability.md:297 calls `trace.get_tracer_provider().add_span_processor()`, which fails on the default proxy provider. Note that `configure_otel_providers()` must run first.
- Signature listings tagged `python` are not valid Python (73 blocks). Tag them `text`, or write them as `def …: ...` stubs, so readers and checkers do not treat them as runnable.
- A few signature listings omit newer parameters:
  - `AgentModeProvider(expose_mode_get, expose_mode_set)`: python/…_comprehensive_guide.md:2412
  - `MemoryContextProvider(index_line_length, index_line_limit)`: :1896
  - `PolicyEnforcementFunctionMiddleware(max_pending_approvals, pending_approval_ttl)`: :3155
  - `A2AAgent(supported_protocol_bindings, forwarded_kwargs)`: microsoft_agent_framework_a2a_protocol.md:30
  - `FunctionTool(result_parser)`: python/…_tools.md:440

  Add the missing parameters.

**Revision history**

- Revision-history tables (python/index.mdx:171–188, python/…_comprehensive_guide.md:3695–3703) are long logs of guide maintenance ("Installed and introspected… Documented eight classes…"). Trim each to a few dated rows.

#### Top 5 improvements

1. **Fix the 20 broken Python examples** (B2–B14) and add the stub-client runner (`scratchpad/maf/pyrun.py`) to CI, so HITL, streaming, routing and A2A snippets cannot regress.
2. **Make every page reachable and every link work.** Convert the 64 `./` links to root-absolute links (B1), list all pages in the sidebar, link the quick-start page, and delete `migration_notice.md` (M2, M3).
3. **Split up the Python comprehensive guide.** Remove the "What's new in 1.14.0" section, the duplicated orchestration and harness sections, the errata notes and the `azure-ai-agents` appendix. Turn the class reference into short tables on the topic pages (M4–M6).
4. **Run one sweep for model IDs and status labels** against today's providers and `__feature_id__`. This covers M1, M7–M9 and B15–B16, and also fixes the A2A story by documenting `A2AExecutor` hosting.
5. **Bring .NET up to the Python depth.** Add Harness, Anthropic and GitHub Copilot providers, DevUI and evaluation; expand the RBAC stub and the advanced page; and replace generic production bullets with hosting guidance (Foundry hosted agents, Functions, durable agents), security (FIDES, Content Safety) and CI evaluation (M10–M12).

---

### Assessment: Amazon Bedrock Agents / AgentCore / Strands guide

Folder: `src/content/docs/amazon-bedrock-agents-guide/` (13 Python-side pages + `index.mdx`, 4 TypeScript pages). Assessed 2026-10-21 against `strands-agents` 1.62.0, `strands-agents-tools` 0.8.9, `bedrock-agentcore` 1.24.2, `boto3` 1.43.117, `@strands-agents/sdk` 1.21.0, `bedrock-agentcore` (npm) 0.4.6 and `@aws/agentcore` CLI 0.31.1, all matching `src/data/versions.json`. Scratch work: `scratchpad/assess-bedrock/`.

#### Scorecard

| # | Criterion | Score | Justification |
|---|-----------|-------|---------------|
| 1 | Correctness of code | 3/5 | Strands, AgentCore, A2A and all 31 TS blocks check out and the quick starts run end to end. But 9 of 111 Python blocks are broken or stubs (guardrail params, `apply_guardrail` enum, `failed_nodes` iteration, production-guide classes that call undefined methods, an invalid CodePipeline definition), and 3 AgentCore CLI commands fail or do the wrong thing. |
| 2 | Currency | 4/5 | Placeholders are used throughout, with one literal-version warning. Every model ID is a current Claude 5.5 Bedrock ID, checked against the AWS model cards. The maintenance-mode date is checked against AWS docs. Some smaller items are dated: Node 18, GitHub v1 source action, `apt-get awscli`. |
| 3 | Accuracy of prose | 3/5 | The rewritten AgentCore, Strands and A2A pages are accurate. The Classic pages still have invented material: a Flows node schema, DynamoDB session-state diagrams, client-side "chunking strategies", and marketing percentages. The getting-started deploy step creates a harness project, not the code shown. |
| 4 | Coverage | 3/5 | Runtime, Memory, Code Interpreter, Identity, A2A and the Strands multi-agent primitives are well covered. Missing: the managed Harness (AWS's recommended migration target), Gateway-with-Strands code, Python Browser, Classic→AgentCore CLI import, and Strands structured output, providers, interrupts, plugins and sandboxes. Agents Classic, which is in maintenance mode, is over-covered. |
| 5 | Structure and navigation | 2/5 | No Python topic page is in the sidebar (only the TS pages are). There is no `python/` folder, unlike `typescript/`. An archived per-release page is linked from the index. Diagrams and production cover only Classic. Links pass (0 broken). |
| 6 | Consistency | 3/5 | 13 pages have no `language`, and the index `framework` value differs from every other page. American spellings and hype remain in the Classic pages. API style is consistent within each SDK. |
| 7 | Production readiness | 2/5 | The production guide covers only Classic and is full of stubs and undefined helpers. AgentCore production is one short section. The TS deployment page is the only solid production material. |

**Verdict.** The AgentCore, Strands, A2A and TypeScript material is now trustworthy: the code compiles, runs against mocks and matches the SDKs. But it sits inside a navigation and production layer still built around Agents Classic. The biggest problems are three AgentCore CLI commands that fail or deploy the wrong thing, broken Classic and production snippets, Python pages that cannot be reached from the sidebar, and no coverage of the Harness that AWS now recommends. Fix the 8 blockers, move the Python pages into `python/` with sidebar entries, and replace the Classic production guide with an AgentCore one. That would raise this guide to a 4.

#### Code verification

Method:
- Python: AST parse, plus import resolution, plus a signature checker (`assess-bedrock/sigcheck.py`). The checker compares keyword arguments with `inspect.signature`, and every boto3 call (top-level and nested literal dicts) with the botocore service model. It also checks required members and enum values.
- Instance-method calls were spot-checked by hand with `inspect` (MemoryClient, CodeInterpreter, GraphBuilder, GraphResult, A2AAgent, MCPClient and others).
- End-to-end runs used a scripted fake Strands `Model` (`assess-bedrock/fakemodel.py`, `e2e.py`).
- TS: `tsc --strict` (NodeNext). The TS minimal example was run with `tsx` against a stub model.
- CLI: `@aws/agentcore@0.31.1`, checked with `--help` and real `create` runs.

| Page | Blocks | Pass | Fail | Notes |
|------|--------|------|------|-------|
| index.mdx | 2 py, 1 bash | 3 | 0 | Minimal `BedrockAgentCoreApp` example served `/invocations` and `/ping` with the fake model |
| bedrock_agentcore_comprehensive_guide.md | 23 py, 2 bash, 1 json | 25 | 1 | All boto3 shapes and enums valid (runtime `PYTHON_3_10`–`3_14`/`NODE_22`, authorizer, protocol, memory strategies). E2E passed for: quick start (incl. `context.session_id` and the error path), async-generator streaming (SSE), and `@app.async_task` (ping → `HealthyBusy`). Bash L176–179 fails (see B1) |
| bedrock_strands_sdk_guide.md | 29 py, 2 bash | 30 | 1 | E2E passed for: tool call, `ToolContext` + `invocation_state`, `as_tool`, conditional-loop graph, swarm handoff, `invoke_async` voting, `stream_async`. Swarm-member `session_manager` restriction confirmed. L568 fails |
| bedrock_a2a_protocol_guide.md | 5 py, 3 bash, 4 json | 11 | 1 | `A2AServer` E2E: the card and the `message/send` task response match the page's JSON. `agentcore create` matches the generated scaffold once `--memory` is added. Bash L274 fails |
| bedrock_agents_advanced_multi_agent_python.md | 8 py | 8 | 0 | `TracePart.collaboratorName` and the `A2AAgent(endpoint=…)` signature confirmed |
| bedrock_agents_comprehensive_guide.md | 13 py, 6 bash, 3 json, 1 yaml, 1 hcl, 1 js | 22 | 3 | Failures: L997 guardrail, L1121 `apply_guardrail`, L1146 invented Flows schema. YAML/HCL not executed |
| bedrock_agents_recipes.md | 9 py | 9 | 0 | Minor race in `prepare_and_alias` (see Minor) |
| bedrock_agents_production_guide.md | 13 py | 8 | 5 | L36 stub health checks; L832 DR, L909 pipeline, L1054 compliance and L1098 troubleshooting call undefined methods or invalid params |
| bedrock_agents_error_handling_python.md | 3 py | 3 | 0 | |
| bedrock_agents_middleware_python.md | 1 py | 1 | 0 | Runs, but has a cache-key security bug (M9) |
| bedrock_agents_observability_python.md | 5 py | 5 | 0 | `invoke_agent_runtime` `traceParent`/`traceId` confirmed |
| bedrock_security_iam_examples.md | 6 json | 6 | 0 | All parse |
| bedrock_agents_diagrams.md, march_2025_updates_summary.md | 0 | – | – | ASCII only |
| typescript/index.mdx | 1 ts, 2 bash | 2 | 1 | `tsc` pass. Ran with a stub model: tool call, JSON response, 400 without a session ID. The inline `agentcore create` at L77 fails |
| typescript/strands_typescript_guide.md | 18 ts | 18 | 0 | Exports checked (`ConcurrentInvocationError`, `StructuredOutputError`, `McpClient.loadServers`, `/a2a`, `/telemetry`, …) |
| typescript/agentcore_typescript_guide.md | 9 ts | 9 | 0 | |
| typescript/agentcore_typescript_production.md | 3 ts, 3 bash, 1 json | 7 | 0 | CLI flags (`dev --stream`, `traces list`, `--build`, `--memory`) confirmed |
| **Total** | **111 py, 31 ts** | **Python 102/111, TS 31/31** | **9 py + 3 CLI** | |

Other checks:
- `node scripts/check-versions.mjs | grep bedrock`: 1 warning (TS production L58).
- `node scripts/check-links.mjs`: 0 broken.
- Stale model IDs: none. All 109 Claude references are `{,us.,global.}anthropic.claude-{opus,sonnet,haiku}-5-5`, checked against the AWS model cards (`docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-*-5-5.html`).
- The Agents Classic maintenance-mode text (30 July 2026) matches AWS's page.

#### Issues

### Blocker

| # | File:line | Problem | Fix |
|---|-----------|---------|-----|
| B1 | bedrock_agentcore_comprehensive_guide.md:176–179 | `agentcore create --name SalesAnalyst --defaults` creates a **harness** project. The CLI prints "Creating a harness project (pass --framework to create an agent project instead)". The `main.py` shown just above is never deployed | Use `agentcore create --name SalesAnalyst --framework Strands --model-provider Bedrock --memory none`, replace `app/SalesAnalyst/main.py` with the example, then `agentcore deploy` and `agentcore invoke "…"`. Mention `--defaults` separately as the harness route |
| B2 | bedrock_a2a_protocol_guide.md:274–275 | The command fails: "provide all: --framework, --model-provider, --memory" | Add `--memory none` |
| B3 | typescript/index.mdx:77 | `agentcore create --language TypeScript --framework Strands` fails for the same reason | Use `agentcore create --name MyAgent --language TypeScript --framework Strands --model-provider Bedrock --memory none` |
| B4 | bedrock_strands_sdk_guide.md:568 | `for node_id in result.failed_nodes:` raises `TypeError`. `GraphResult.failed_nodes` is an `int` (count) | Iterate `node_id, r in result.results.items()` and test `r.status == Status.FAILED`, or print the count |
| B5 | bedrock_agents_comprehensive_guide.md:1050–1066, 1090 | `create_guardrail` fails client-side validation: `wordsConfig[].action` and `managedWordListConfig` are not valid fields, and `BANK_ACCOUNT_NUMBER` is not a PII type | Use `inputAction`/`outputAction` (or omit them), `managedWordListsConfig`, and `US_BANK_ACCOUNT_NUMBER` or `INTERNATIONAL_BANK_ACCOUNT_NUMBER` |
| B6 | bedrock_agents_comprehensive_guide.md:1124 | `apply_guardrail(source='USER_INPUT')`: the valid values are `INPUT`/`OUTPUT`, so this returns ValidationException | Use `source='INPUT'` |
| B7 | bedrock_agents_production_guide.md:926–981, 987–1031 | The `create_pipeline` dict fails validation (`artifacts` should be `artifactStore`). Owner `GitHub` is not a valid `ActionOwner`. The Deploy action has no configuration or artifacts. `buildspec` is defined but never passed to `create_project` | Use `artifactStore`, a `CodeStarSourceConnection` source with `outputArtifacts`, and full CloudFormation action config. Pass `source={'type': 'GITHUB', 'location': …, 'buildspec': buildspec}`. Or replace with a short CDK/`agentcore deploy` pipeline |
| B8 | bedrock_agents_production_guide.md:114, 203–211, 870–873, 885–888, 928, 1024, 1082–1086, 1106–1110 | Classes call methods that are never defined: `_replicate_knowledge_bases`, `_setup_route53_failover`, `_test_*`, `_get_pipeline_role_arn`, `_get_codebuild_role_arn`, `_check_*`, `_diagnose_*`. That is an `AttributeError` at runtime. Health checks are stubs that always return `'PASS'`, and `'health_check': 'PASSED'` is hard-coded | Implement them, or replace the classes with prose plus one working check each (e.g. a real `invoke_agent` smoke test). Remove the hard-coded PASS values |

### Major

| # | File:line | Problem | Fix |
|---|-----------|---------|-----|
| M1 | astro.config.mjs:204–219 | Only "Overview (Python)" and the TS pages are in the sidebar. None of the 13 Python topic pages is listed, so they are reachable only through index cards | Add a "Python" group listing AgentCore, Strands, A2A, multi-agent, recipes, production, observability, error handling, middleware, IAM and diagrams |
| M2 | folder layout | Python pages sit at the guide root while TS pages are in `typescript/`. AGENTS.md expects `python/` when there are several languages | Move them to `python/`, add redirects, and update links and the sidebar |
| M3 | march_2025_updates_summary.md (whole), index.mdx:65, 83 | An archived per-release changelog (AGENTS.md says "do not create per-release pages") full of false claims, emojis and "100% complete", linked twice from the index | Delete the page and its index cards. Fold the corrections into index "What's new" |
| M4 | bedrock_agents_comprehensive_guide.md:95–120, 124–231 | The trust policy trusts bedrock, lambda, states, events and the account root on one role. The permissions mix caller actions (`CreateAgent`…) with service-role actions. `Resource: arn:aws:bedrock:*:ACCOUNT_ID:*` does not match foundation-model ARNs (no account), so model invocation is denied. This contradicts the least-privilege IAM page | Replace with the scoped trust and service-role policies from bedrock_security_iam_examples.md:24–60, and a separate caller policy |
| M5 | bedrock_agents_diagrams.md:577–670 (also 239–282, 523–573) | Shows Bedrock Agents session state "stored in DynamoDB, TTL 24 hours" and a four-layer memory in DynamoDB/RDS. That is invented: Classic manages sessions internally (`idleSessionTTLInSeconds`, `memoryConfiguration`). The Flows diagram does not reflect real Flows nodes. There are no AgentCore or Strands diagrams | Redraw session/memory from the real features. Add Runtime/Memory/Gateway and Strands graph/swarm diagrams, and consider Mermaid |
| M6 | bedrock_agentcore_comprehensive_guide.md:64, 691; index.mdx:100 | The managed Harness is named as AWS's recommended migration target but never shown. `create_harness`/`invoke_harness` and `agentcore create --defaults` exist | Add a Harness section with a boto3 `create_harness` example and the CLI route |
| M7 | bedrock_agentcore_comprehensive_guide.md:689–691 | The Classic migration section omits the CLI import path (`agentcore create --type import --agent-id … --agent-alias-id … --region …`) and `agentcore export` (harness → Strands) | Add both, with a short walk-through |
| M8 | bedrock_agents_production_guide.md (whole) | Production guidance covers only Classic (maintenance mode). AgentCore production is a single section (agentcore guide L860–908): no quotas or scaling, Evaluations, CI/CD with `agentcore deploy`, JWT inbound auth testing, or cost per session | Write an AgentCore production page (CLI/CDK pipeline, endpoints/versions, Observability, Evaluations, quotas, cost) and cut the Classic one to essentials |
| M9 | bedrock_agents_middleware_python.md:82–85, 101 | The cache key is only the question text, while the agent receives a per-user `userId`. One user's account-specific answer is served to another | Include `user_id` (and `session_id`) in the key, or cache only explicitly stateless intents |
| M10 | bedrock_agents_comprehensive_guide.md:1144–1269 | `PromptFlowManager` returns an invented node schema (`INVOKE_MODEL`, `CONDITIONAL`, `INVOKE_AGENT`) that `create_flow` would reject | Replace with a minimal real `create_flow` definition (Input → Prompt → Output) plus `prepare_flow`/`invoke_flow` |
| M11 | bedrock_agents_comprehensive_guide.md:901–977 | `ChunkingStrategyManager` is local Python chunking unrelated to Bedrock KBs, which chunk server-side (`FIXED_SIZE`, `HIERARCHICAL`, `SEMANTIC`, `NONE`) | Replace with `chunkingConfiguration` examples for each strategy |
| M12 | bedrock_strands_sdk_guide.md (coverage); index.mdx:101 | Missing: structured output (`structured_output_model`), non-Bedrock providers (Anthropic, OpenAI, LiteLLM), interrupts (prose only, L557), plugins and skills, sandboxes, MCP over HTTP to Gateway, and bidi streaming. The index advertises sandboxes, plugins and skills | Add short sections with verified code for each |
| M13 | all 13 Python pages (frontmatter); index.mdx:4 | `language:` is missing everywhere except the index. The index uses `framework: amazon-bedrock` while every other page uses `amazon-bedrock-agents` | Add `language: python` and pick one framework key |
| M14 | bedrock_agents_comprehensive_guide.md (coverage) | For existing Classic users: no return-of-control loop (`returnControl` event → `returnControlInvocationResults`), no `memoryConfiguration`, no prompt overrides, no `invoke_inline_agent`, no `retrieve_and_generate`. Meanwhile 1,883 lines are spent on generic classes | Replace the generic classes (ModelSelectionFramework, MultiRegionAgentDeployment, IAM manager) with these features |

### Minor

| # | File:line | Problem | Fix |
|---|-----------|---------|-----|
| m1 | typescript/agentcore_typescript_production.md:58 | Literal current version "CLI 0.31.1" (check-versions warning) | `%%version=@aws/agentcore%%`, or "in the current CLI" |
| m2 | bedrock_strands_sdk_guide.md:117; bedrock_agents_comprehensive_guide.md:1277 | Lists `jp.` as a prefix for these IDs, but Sonnet 5.5 has no `jp.` profile (model card) | Say prefixes vary by model and point to the card |
| m3 | index.mdx:153–157; bedrock_strands_sdk_guide.md:945 | Revision history is out of order, and "November 2025 – Strands ~1.0" and "1.0 – March 2025" contradict 1.0 = July 2025 | Sort the rows and fix the dates or labels |
| m4 | index.mdx:40–83 | Relative `./` links; AGENTS.md asks for root-absolute | Use `/amazon-bedrock-agents-guide/...` |
| m5 | index.mdx:45 | "Local/OSS Python + TypeScript SDK", but the page is Python only | Say "Python SDK" |
| m6 | index.mdx:40–50 | Zero → Hero teaches AgentCore hosting (steps 1–5) before Strands, which you need to write the agent (step 6) | Put Strands first |
| m7 | index.mdx:49 | The card says "IAM roles, policies, VPC", but the IAM page has no VPC content | Add VPC endpoint/Runtime VPC content, or change the text |
| m8 | bedrock_agents_comprehensive_guide.md:57, 59, 63; bedrock_agents_error_handling_python.md:65; bedrock_agents_production_guide.md:640, 1040, 1043 | American spellings (specialized, organizational, optimize, analyze, behavior) | British English |
| m9 | bedrock_agents_comprehensive_guide.md:13; bedrock_agents_production_guide.md:3, 7, 1154; bedrock_agents_diagrams.md:3, 7 | Hype ("definitive", "exhaustive", "enterprise scale", "comprehensive") | Plain descriptions |
| m10 | bedrock_agents_production_guide.md:642–643 | Unsourced "up to 90% / 85% / 30%" | Cite the AWS pages or drop the numbers |
| m11 | bedrock_agents_production_guide.md:722–730 | The "AgentCore Observability" heading is followed by a Classic note and a "Bedrock Agents CloudWatch" bullet, which is confusing | Split into Classic and AgentCore subsections, or link to the observability page |
| m12 | bedrock_agents_comprehensive_guide.md:75 | "Node.js 18+" (EOL) | Node.js 22 |
| m13 | bedrock_agents_comprehensive_guide.md:1409 | `apt-get install awscli` installs CLI v1 | Remove it, or note v1 |
| m14 | bedrock_agents_comprehensive_guide.md:1576 | The "Multi-Agent Systems (MAS)" heading covers only console/CFN/Terraform agent creation | Rename to "Creating agents (console, CloudFormation, Terraform)" |
| m15 | bedrock_agents_comprehensive_guide.md:1096–1097 | Regex named `AWS_SECRET_KEY` matches access-key IDs (`AKIA…`) | Rename it to `AWS_ACCESS_KEY_ID`, or use the built-in `AWS_ACCESS_KEY` PII type |
| m16 | bedrock_agents_recipes.md:36–42 | `prepare_and_alias` can see a stale `PREPARED` on re-prepare, and does not wait for the alias to be `PREPARED` before it is used as a collaborator | Wait for `PREPARING` → `PREPARED`, then poll `get_agent_alias` until `PREPARED` |
| m17 | bedrock_agents_recipes.md:617 | `sessionId=f"doc-{key…}"` can contain characters outside `[0-9a-zA-Z._:-]` | Hash the key |
| m18 | bedrock_agents_error_handling_python.md:39, 63–65 | The retry set omits `ModelNotReadyException`/`ServiceQuotaExceededException`. The DLQ section is prose only | Add them; give a DLQ example or drop it |
| m19 | bedrock_strands_sdk_guide.md:486 | Prints `GraphNode` objects | `[n.node_id for n in result.execution_order]` |
| m20 | bedrock_agents_middleware_python.md:7; bedrock_agents_error_handling_python.md:7; bedrock_agents_diagrams.md:7; bedrock_agents_production_guide.md:7 | The first paragraph repeats the frontmatter description | Remove the duplicate |
| m21 | bedrock_agents_middleware_python.md:42 | "Configure the API Gateway to forward requests to the Bedrock Agent", but the Lambda does the invoking | Reword |
| m22 | bedrock_agentcore_comprehensive_guide.md:851–854; bedrock_agents_observability_python.md:28–31 | The same ADOT Dockerfile snippet appears on two pages | Keep it on the observability page and link to it |
| m23 | bedrock_agents_diagrams.md:30 | Misaligned box border | Fix the padding |

#### Top 5 improvements

1. **Fix the deploy path end to end** (B1–B3, M6, M7). Make every `agentcore create` command run as written: `--framework … --model-provider … --memory …`, with `--defaults` explained as the harness route. Add a Harness section, and a Classic→AgentCore migration using `agentcore create --type import` and `agentcore export`.
2. **Restructure navigation** (M1–M3). Move the Python pages to `python/` with a full sidebar group like the other guides. Delete the March 2025 page. Order Zero → Hero as Strands → Runtime → Memory/Gateway/Identity → multi-agent → production.
3. **Replace the Classic production guide with an AgentCore production page** (B7, B8, M8). Cover `agentcore deploy`/CDK CI, endpoints and versions, JWT auth, Observability and Evaluations, quotas and cost. Cut the Classic material to working snippets only (no stubs or undefined helpers).
4. **Repair or remove invented and broken Classic content** (B5, B6, M4, M5, M10, M11). That means the guardrail params, `apply_guardrail`, the IAM trust and permissions, the Flows schema, client-side chunking and the DynamoDB session diagrams. Use the space for return of control, memory, prompt overrides and inline agents (M14).
5. **Fill the Strands coverage gaps and normalise the pages** (B4, M12, M13). Add structured output, other providers, interrupts, plugins/skills, sandboxes and MCP to Gateway. Fix `failed_nodes`. Add `language: python` and one `framework` key, convert to British English, and remove the hype.

---

### LlamaIndex guide assessment (2026-10-21)

Scope: `src/content/docs/llamaindex-guide/` — root `index.mdx`, `python/` (9 pages), `typescript/` (4 pages; upstream project deprecated). Read in reading order: overview, Python index, comprehensive guide, advanced agents, advanced implementations, observability, streaming server, recipes, production, diagrams; then the TypeScript index, comprehensive guide, recipes and production guide.

Environment: `llama-index` / `llama-index-core` 0.14.25 (matches `versions.json` and PyPI latest, 2026-09-21), `llama-index-workflows` 2.26.0, Python 3.13 venv; npm `llamaindex` 0.12.1, `@llamaindex/workflow` 1.1.25, `@llamaindex/core` 0.6.23, `@llamaindex/openai` 0.4.23, TypeScript 5.9.3, Node 22. Scratch work: `scratchpad/assess-llamaindex/` (`extract.py`, `pycheck.py`, `pyrun.py`, `claims*.py`, `prodtest.py`, `anchors.mjs`, `ts/`).

#### Scorecard

| # | Criterion | Score | Justification |
|---|---|---|---|
| 1 | Correctness of code | 3 | The core Python pages are solid: 100 of 116 blocks run against 0.14.25 with mock models, and all 40 TS blocks pass `tsc --strict`. Two `SubQuestionQueryEngine` examples fail on a plain install. The Python production app is broken once its blocks are put together, and the TS production code fails its own `tsconfig.prod.json`. |
| 2 | Currency | 3 | `check-versions` is clean, but TS versions are hard-coded as literals (`@llamaindex/workflow` 1.1.25 ×9). Model IDs are stale (`claude-sonnet-4-5`, `gemini-2.5-flash`, `gpt-5-mini` ×52). The guide still names the deprecated `llama-cloud-services`, says Node 20 although it is EOL, and uses old GitHub Actions. |
| 3 | Accuracy of prose | 4 | The claims I spot-checked hold: legacy APIs were removed in 0.13.0, memory is dropped when a `Context` is serialised, the changelog entries for 0.14.22–0.14.25 are right, and the npm deprecation notices are real. The false parts: the root index says both guides cover the same ground, one diagram contradicts the code, and a framework card tagline makes a claim nobody can check. |
| 4 | Coverage | 3 | Agents, workflows, memory, tools, retrieval and evaluation are well covered for Python. Missing: multimodal (new in 0.14.22/0.14.23), a `PropertyGraphIndex` example, LlamaParse/LlamaCloud, `WorkflowServer`, and evaluation datasets. The TS guide has no RAG primitives at all. Diagrams and the production infrastructure sections are padded. |
| 5 | Structure and navigation | 3 | Good index pages (install, working minimal example, page map), and every link and all 118 anchors resolve. But only 3 of 14 pages are in the sidebar, there is a "deep dive" heading, and the Zero → Hero numbering does not match the section numbers. |
| 6 | Consistency | 3 | API style and terminology are consistent across Python pages. But revision tables are uneven (7 of 14 pages have one, and some rows have 4 cells), headings use American spellings, recipes and production repeat their description as the first line, and description case varies. |
| 7 | Production readiness | 2 | The Python production page is mostly generic FastAPI/K8s/JWT/sharding scaffolding, its assembled app does not work, and the CI is outdated. The LlamaIndex-specific advice (observability, `Memory` backed by a database, ingestion as a separate job) is good but short. The TS production page is generic and does not build under its own config. |

**Overall verdict.** Read on its own, the Python comprehensive guide is accurate and current. The advanced-agents, advanced-implementations, observability and streaming pages are also good, and almost every snippet in them runs against 0.14.25. The production guide pulls the quality down: it is not specific to LlamaIndex and is broken once assembled, and the TS production guide fails its own build config. There are also stale model IDs, TS versions hard-coded outside the registry, and 11 pages missing from the sidebar. Fix the 5 blockers and the sidebar, refresh the model IDs, and the Python guide would be a 4.

#### Code verification

How I checked: I extracted every fenced block. For Python, each block was compiled (with top-level await allowed), every import was resolved, keyword arguments were checked statically against `inspect.signature` and pydantic fields, and then each block was executed in sequence per page. `OpenAI` and `OpenAIEmbedding` were swapped for `MockFunctionCallingLLM` and `MockEmbedding`, `input()` was patched, and sample `data/` and `docs/` directories were seeded. For TypeScript, each block was compiled with `tsc --strict` using the guide's own tsconfig (lib ES2022, bundler resolution). Fragments got the imports or definitions from the preceding block on the same page. I ran 11 blocks that need no LLM with Node; their output matched the documented comments, including the "4 events" claim at `typescript/llamaindex_workflows_typescript_comprehensive_guide.md:218`. I also compiled the production `src/**` files with the guide's `tsconfig.prod.json` settings.

### Python (116 blocks)

| Page | Blocks | Pass | Genuine fail | Fragment | Mock-limited* | Env-only** |
|---|---|---|---|---|---|---|
| python/index.mdx | 1 | 1 | 0 | 0 | 0 | 0 |
| llamaindex_comprehensive_guide.md | 65 | 54 | 1 (L1344) | 2 (L535, L1021) | 4 (L748, L769, L796, L1319) | 4 (L412, L442, L1190, L1236) |
| llamaindex_advanced_agents_python.md | 9 | 9 | 0 | 0 | 0 | 0 |
| llamaindex_advanced_implementations.md | 6 | 6 | 0 | 0 | 0 | 0 |
| llamaindex_observability_python.md | 5 | 5 | 0 | 0 | 0 | 0 |
| llamaindex_streaming_server_fastapi.md | 2 | 2 | 0 | 0 | 0 | 0 |
| llamaindex_recipes.md | 10 | 5 | 1 (L349) | 0 | 2 (L33, L549) | 2 (L251, L642) |
| llamaindex_production_guide.md | 18 | 18 (each block runs alone) | Assembled app: 3 logic failures (see B2–B4) | 0 | 0 | 0 |
| **Total** | **116** | **100** | **2 + production app** | **2** | **6** | **6** |

\* The mock LLM cannot produce valid structured output or selector JSON, or its context window is too small. The APIs used were confirmed to exist (`AgentOutput.get_pydantic_model`, `structured_response`, `as_structured_llm`, `LLMSingleSelector`).
\** Needs a missing optional dependency (`sentence_transformers`, `lxml`), an API key at construction (`GoogleGenAI`), a running MCP server, a `./myproject` folder or a SQLite file. These are not documentation errors.

The static signature check found no bad keyword arguments; the `Document(text=)` hits were false positives, since `text` is a pydantic alias. I also ran these claims directly against the installed release:
- the default `as_chat_engine()` returns `CondensePlusContextChatEngine` — true;
- `ctx.to_dict()` emits `UnserializableKeyWarning`, and the restored context has no `memory` — true;
- `asyncio.run(agent.run(...))` raises `RuntimeError: no running event loop` — true;
- the durable `ApprovalFlow` start/cancel/resume (advanced agents §4.2) works end to end and returns `Done: …`;
- a tool exception comes back to the LLM as `is_error=True` output — true;
- `partial_params` hides the bound parameter from the schema — true;
- `allow_parallel_tool_calls` defaults to `True` — true;
- `max_iterations` and `early_stopping_method` are real `run()` parameters;
- `AgentWorkflow` is not importable from `llama_index.core.workflow` — true;
- `workflow_as_mcp` exists, and `react_header` is a prompt key.

### TypeScript (40 blocks)

| Page | Blocks | `tsc --strict` pass | Run with Node |
|---|---|---|---|
| typescript/index.mdx | 1 | 1 | 1/1 output as documented |
| llamaindex_workflows_typescript_comprehensive_guide.md | 18 | 18 (6 fragments given context) | 8/8 output as documented |
| llamaindex_typescript_recipes.md | 10 | 10 | 1/1 (Recipe 9) |
| llamaindex_typescript_production_guide.md | 11 | 11 with the guide's dev tsconfig; **4 files fail** with its own `tsconfig.prod.json` (B5) | — |
| **Total** | **40** | **40 (strict) / 36 under prod config** | **11/11** |

`node scripts/check-versions.mjs | grep llamaindex`: clean. It does not catch the TS literals, because `@llamaindex/workflow`, `@llamaindex/core` and `@llamaindex/openai` are not in the registry. `node scripts/check-links.mjs`: 0 broken. My own anchor check (`anchors.mjs`, using github-slugger) checked 118 in-guide `#anchor` links: 0 broken.

#### Issues

### Blocker

| # | File:line | Problem | Fix |
|---|---|---|---|
| B1 | python/llamaindex_comprehensive_guide.md:1347; python/llamaindex_recipes.md:403 | `SubQuestionQueryEngine.from_defaults(...)` raises `ImportError: llama-index-question-gen-openai package cannot be found` after the documented `pip install llama-index`. The bundle only pulls in core plus OpenAI LLM/embeddings plus nltk. | Add `pip install llama-index-question-gen-openai` beside both examples, or pass `question_gen=LLMQuestionGenerator.from_defaults(llm=llm)` (from `llama_index.core.question_gen`). |
| B2 | python/llamaindex_production_guide.md:721, 807, 825 | `JSONResponse` is used in three exception handlers but never imported, so the global handler itself raises `NameError`. | Add `from fastapi.responses import JSONResponse` to the block at line 661 or 74. |
| B3 | python/llamaindex_production_guide.md:608, 700, 727 | Three handlers are registered on `POST /query`. FastAPI routes to the first, the unauthenticated placeholder `query_endpoint`, so the JWT-protected route (700) and the rate-limited route (727) are dead code. Auth and rate limiting are silently bypassed. | Keep one `/query` route that combines `Depends(get_current_user)`, `@limiter.limit(...)` and the tracing spans, and calls the real query engine instead of returning `{"result": "response"}`. |
| B4 | python/llamaindex_production_guide.md:205-206, 321-334, 1144-1156 | `/health` and `/ready` are never defined, but the Docker `HEALTHCHECK` and the K8s liveness/readiness probes rely on them, so pods never become ready. The integration test posts to `/documents`, which does not exist, and sends a JSON body to an endpoint that takes `query` as a query parameter. Assembled and run, it fails its assertion. | Add `@app.get("/health")` and `@app.get("/ready")` (the latter checking that the index loaded), define a `QueryRequest` pydantic body model, and either add `/documents` or drop it from the test. |
| B5 | typescript/llamaindex_typescript_production_guide.md:345, 349, 398, 408, 533, 535-547 | The guide's own `tsconfig.prod.json` (NodeNext, `noUnusedParameters`, `noImplicitReturns`) fails `npm run build` on its own code. Under NodeNext the `import Redis from 'ioredis'` default import is not constructable (TS2351/TS2709). The unused `ctx` handler parameters fail TS6133, as does the unused `z` import, and `validate()` fails TS7030. | Use `import { Redis } from 'ioredis'`, rename unused handler parameters to `_ctx`, drop `z` from the import, and `return next()`. Or relax the two `noUnused*` flags in 1.2. |

### Major

| # | File:line | Problem | Fix |
|---|---|---|---|
| M1 | python/llamaindex_comprehensive_guide.md:419 | `claude-sonnet-4-5` is a superseded model ID. | Use `claude-sonnet-5-5`. |
| M2 | python/llamaindex_comprehensive_guide.md:420 | `gemini-2.5-flash` is stale; the installed `llama-index-llms-google-genai` knows `gemini-3.7-flash`. | Use the current Flash model (`gemini-3.7-flash`). |
| M3 | 52 occurrences across 9 pages, e.g. python/index.mdx:109, python/llamaindex_comprehensive_guide.md:344, typescript/llamaindex_typescript_production_guide.md:99,117; `gpt-5-nano` at python/llamaindex_production_guide.md:1011,1069 | `gpt-5-mini` / `gpt-5-nano` date from August 2025. The current `llama-index-llms-openai` model table lists the gpt-5.4/5.5 and gpt-6 families. | Switch to a current small OpenAI model throughout, and keep the existing "model names are examples" note (comprehensive guide :355). |
| M4 | python/llamaindex_comprehensive_guide.md:1220 | Recommends LlamaParse via `llama-cloud-services`. That package is deprecated (its README says it was maintained only until 1 May 2026; last release February 2026). | Point to the `llama-cloud` SDK (2.19.0) and the current LlamaParse docs. Keep LiteParse (`liteparse`, real and current). |
| M5 | typescript/index.mdx:22,26; typescript/llamaindex_workflows_typescript_comprehensive_guide.md:8,10,12; typescript/llamaindex_typescript_recipes.md:8,14; typescript/llamaindex_typescript_production_guide.md:8,144-146 | The current versions of `@llamaindex/workflow` (1.1.25), `@llamaindex/core` (0.6.23) and `@llamaindex/openai` (0.4.23) are written as literals, against AGENTS.md "one source of truth". `check-versions` misses them because the keys are not registered. | Add the three packages to `src/data/versions.json` with `guide: llamaindex-guide` and use `%%version=@llamaindex/workflow%%` and the like (in JSON blocks too). |
| M6 | astro.config.mjs:222-228 | The LlamaIndex sidebar has only Overview/Python/TypeScript. The other 11 pages (comprehensive, advanced ×2, observability, streaming, recipes, production, diagrams; three TS pages) are reachable only through index cards. LangGraph, for example, has a "Reference" group. | Add collapsed Python and TypeScript groups listing every page. |
| M7 | index.mdx:31-52 | "What you'll find in **both** guides" lists data ingestion, "Summary, tree, vector store, list indices", query engines (subquestion/router), ReActAgent and CodeActAgent. The TS guide covers none of these; it covers workflows and `FunctionAgent`/`multiAgent` only. "List index" is the old name, and the Python guide says to use `SummaryIndex`. | Re-title the section "Python guide" or split the cards per language; drop "list"/"tree". |
| M8 | python/llamaindex_production_guide.md:1036, 1063-1085 | The cost section does not work. `PRICING_PER_1K_TOKENS` is undefined (`NameError`), `OpenAI` is not imported in the block at 1056, and `CostOptimizedQueryEngine` never calls `log_usage`, so `total_cost` is always 0 and the budget check never fires. | Feed `TokenCountingHandler` (or `LLMChatEndEvent.response.raw.usage`) into `log_usage`, define the pricing dict from config, and import `OpenAI`. |
| M9 | python/llamaindex_production_guide.md:1241, 1250, 1258, 1271-1276, 1284, 1225 | The CI is outdated or broken. It uses `codecov-action@v3`, `trivy-action@master` (unpinned), `codeql-action/upload-sarif@v2` (retired) and `build-push-action@v4` with no `docker/login-action`. `KUBECONFIG` is given the secret's *content* (it expects a path), and `requirements-test.txt` is never defined. | Bump to current majors (codecov v5, upload-sarif v3, build-push v6), pin trivy to a tag, add login/setup-buildx, write kubeconfig to a file, define test requirements. |
| M10 | python/llamaindex_production_guide.md:215-270 vs 119-128 | The compose file runs Chroma, Postgres and Redis while the app uses Pinecone. The Chroma service uses `CHROMA_DB_IMPL=duckdb`, removed in Chroma 0.4, and the pre-1.x data path `/chroma/chroma`. | Make compose match the app (Pinecone needs no service; or switch the app to Chroma/pgvector), and drop the obsolete env var. |
| M11 | typescript/index.mdx:26; typescript/llamaindex_workflows_typescript_comprehensive_guide.md:8; typescript/llamaindex_typescript_production_guide.md:9, 134, 157, 761 | "Node 20+" and `engines >=20` / `@types/node ^20`: Node 20 reached end of life on 30 April 2026. The revision note at production :761 calls Node 18 EOL but moves to another EOL line. | State Node 22+ (and 24 LTS); bump `@types/node`. |
| M12 | python/llamaindex_comprehensive_guide.md:1058-1064 | The LlamaAgents section is thin and partly wrong. `llamactl serve` is shown straight after `init` (you must `cd` into the scaffolded app, and cloud deploys need `llamactl auth login`). `WorkflowClient` needs `pip install 'llama-index-workflows[client]'`, and the built-in `workflows.server.WorkflowServer` is not mentioned. | Add `cd my-app`, the auth step, the `[client]` extra and a short `WorkflowServer` example. |
| M13 | coverage: python/llamaindex_comprehensive_guide.md (whole) | Missing: multimodal documents, query engines and synthesis (headline features of 0.14.22/0.14.23); a runnable `PropertyGraphIndex` example (only named at :229 and recipes :881); LlamaParse/LlamaExtract; workflow resources/DI and `workflows.testing`; `RagDatasetGenerator` / `RetrieverEvaluator` code (named only at :1432). | Add short sections for multimodal, property graph and evaluation datasets; link to the LlamaCloud docs for parsing. |
| M14 | typescript/* (whole) | The TS guide documents only the workflow runtime. Recipe 1 "Basic RAG Chatbot" (typescript/llamaindex_typescript_recipes.md:34-92) stubs retrieval and has no chat history, and no page uses `VectorStoreIndex`, readers or query engines from the `llamaindex` package. | Either add one real `llamaindex` RAG example (index + retriever) or rename the recipes to make clear retrieval is a stub. |
| M15 | src/data/frameworks.ts:127, 139 | The tagline "RAG + AgentWorkflow 1.0" invents a version (AgentWorkflow has no 1.0 release; it ships inside core 0.14.x). The TS tagline hard-codes "final release 0.12.1". | Use "RAG + workflow-based agents", and leave the version to the `version: v('llamaindex')` field. |

### Minor

| # | File:line | Problem | Fix |
|---|---|---|---|
| m1 | python/llamaindex_comprehensive_guide.md:23, 151 | "Core concepts deep dive" breaks the no-"deep dive" rule; the anchor is also linked from python/index.mdx:38, 55. | Rename to "Core concepts" and update both links. |
| m2 | python/index.mdx:45-47 | The Zero → Hero numbering drifts from the section numbers (step 10 → §11, 11 → §12, 12 → §13), and §10 and §14–22 are skipped. | Number the cards by section, or drop the numbers. |
| m3 | python/llamaindex_comprehensive_guide.md:538 | `FunctionAgent(tools=[...], ...)` raises if copied (`'ellipsis' object has no attribute '__name__'`). | Use `tools=[multiply, add]`. |
| m4 | python/llamaindex_comprehensive_guide.md:1025-1028 | A `@step` method shown at module level raises `WorkflowValidationError` if copied. | Show it inside `class ProgressFlow(Workflow):`. |
| m5 | python/llamaindex_diagrams.md:250-251 | The hybrid search diagram says "Weighted Combination", but the guide's code (comprehensive :1280) uses reciprocal-rank fusion. | Say "Reciprocal-rank fusion (QueryFusionRetriever)". |
| m6 | python/llamaindex_diagrams.md:513-523, 598-619 | TreeIndex and KeywordTableIndex get equal billing although the comprehensive guide (:229) calls them rarely right. The "Memory Types" diagram is generic and does not match the `Memory` class. | Trim to the indexes and memory classes the guide actually recommends. |
| m7 | python/llamaindex_recipes.md:8; python/llamaindex_production_guide.md:8 | The first body line repeats the frontmatter description. | Delete it. |
| m8 | python/llamaindex_production_guide.md:3; python/llamaindex_diagrams.md:3 | Descriptions are Title Case or padded ("Comprehensive visual documentation…"). | Use a plain sentence-case description. |
| m9 | python/llamaindex_recipes.md:17, 130, 450; python/llamaindex_production_guide.md:390, 392, 996, 1054; python/llamaindex_diagrams.md:743, 745 | American spellings in headings: Analyzer, Analyzing, Optimization. | Use Analyser, Analysing, Optimisation. |
| m10 | index.mdx:62; typescript/index.mdx:110; typescript/llamaindex_workflows_typescript_comprehensive_guide.md:785; typescript/llamaindex_typescript_recipes.md:662; typescript/llamaindex_typescript_production_guide.md:762 | Revision rows have a 4th cell ("Claude routine") in 3-column tables. 7 Python pages have no revision table at all. | Drop the extra cell; either add tables everywhere or keep them only on index pages. |
| m11 | python/llamaindex_production_guide.md:611, 703 | The production code logs and traces raw user queries (`span.set_attribute("query", …)`, `logger.info(..., query=query)`), against the page's own advice (observability :112-114). | Log a hash or length, or redact. |
| m12 | python/llamaindex_production_guide.md:955-991 | `DistributedCache` has `async` methods that call the blocking `redis` client, which stalls the event loop. | Use `redis.asyncio`. |
| m13 | python/llamaindex_production_guide.md:206 | `HEALTHCHECK` uses `requests`, which is not in requirements, and passes on 404. | Use `curl -f` or `urllib` with a status check. |
| m14 | python/llamaindex_production_guide.md:1165-1179 | The performance test asserts a 0.5 s average against a real LLM-backed engine, which is unrealistic. | Use mock models for the latency test, or set realistic bounds. |
| m15 | python/llamaindex_production_guide.md:900 | `server_name api.llamaindex.com` uses a real third-party domain. | Use `api.example.com`. |
| m16 | python/llamaindex_recipes.md:1062-1066 | The triage recipe wraps sync `llm.complete` in `asyncio.to_thread` when `acomplete` exists. | `await asyncio.gather(self.llm.acomplete(...), ...)`. |
| m17 | python/llamaindex_observability_python.md:83 | `tiktoken.encoding_for_model("gpt-4o")` is an old example model. | Use `tiktoken.get_encoding("o200k_base")`. |
| m18 | typescript/llamaindex_typescript_production_guide.md:207, 227 | Compose uses the obsolete `version: '3.8'`, and its healthcheck calls `curl`, which `node:22-alpine` does not ship, so the container is always unhealthy. | Drop `version`; reuse the Dockerfile's `node -e` check. |
| m19 | typescript/llamaindex_typescript_production_guide.md:481 vs 143-155; 140, 162 | `prom-client` is imported but missing from dependencies. Jest with ts-jest under `"type": "module"` needs an ESM preset that is not shown. `helmet`, `cors` and `dotenv` are listed but never used. | Add `prom-client`, show `jest.config` (ts-jest ESM) or use Vitest, and remove the unused dependencies. |
| m20 | typescript/llamaindex_typescript_production_guide.md:747-752 | `docker push` runs without a registry login; `REGISTRY_TOKEN` is set but unused. | Add `docker/login-action`. |
| m21 | typescript/llamaindex_workflows_typescript_comprehensive_guide.md:719-722 | The orchestrator sends `researchTaskEv` twice for high priority, so the work is duplicated, yet the output reports `tasksCompleted: 1`. | Send once, or explain the fan-out and collect both. |
| m22 | typescript/llamaindex_typescript_recipes.md:384, 592 | `or` is imported but unused (fails `noUnusedLocals`). | Remove the import. |
| m23 | typescript/llamaindex_workflows_typescript_comprehensive_guide.md:774-775 | "Upstream documentation" lists local `node_modules/...` paths. | Link to the npm/GitHub source instead. |
| m24 | python/llamaindex_streaming_server_fastapi.md:72 | The `finally: await handler.cancel_run()` advice for client disconnects is given only in prose. | Show the `try/finally` in the agent SSE code. |

#### Top 5 improvements

1. **Rebuild the Python production guide around a single working LlamaIndex app.** One `/query` route with auth and rate limiting that calls the real engine, real `/health` and `/ready`, imports fixed, cost tracking fed from `TokenCountingHandler`, compose matching the app, current CI actions. Cut the generic sharding, nginx and Fernet material in favour of LlamaIndex-specific concerns: an ingestion job with `IngestionPipeline` + docstore upserts, `Memory` on Postgres, workflow timeouts, prompt-injection and tool-sandboxing guidance (fixes B2–B4, M8–M10).
2. **Fix the two `SubQuestionQueryEngine` examples and the TS production build** (B1, B5). These are the only places where the documented code fails on a clean install.
3. **Refresh model IDs and stale references in one sweep.** `claude-sonnet-5-5`, the current Gemini Flash, a current OpenAI small model; `llama-cloud` instead of `llama-cloud-services`; Node 22+ (M1–M4, M11).
4. **Put every LlamaIndex page in the sidebar and register the TS packages in `versions.json`.** This replaces the nine literal version strings with placeholders and corrects the "both guides" card section (M5–M7).
5. **Fill the Python coverage gaps that matter in 0.14.x.** Multimodal RAG, a runnable `PropertyGraphIndex` example, serving with `WorkflowServer`/`llamactl` (with the `[client]` extra), and evaluation datasets. In return, trim the generic ASCII diagrams and the stub-tool recipes (M12, M13, m5, m6).

---

### Semantic Kernel guide: assessment (2026-10-21)

Scope: `src/content/docs/semantic-kernel-guide/` (18 pages: 5 root, 8 Python, 5 .NET). Read in order: index, concepts, recipes, production and diagrams (root), then Python, then .NET.
Versions under test: `semantic-kernel` 1.45.0 (PyPI, released 2026-10-06), `Microsoft.SemanticKernel` 1.81.0 (NuGet), plus `agent-framework-core` 1.22.0 for the migration bridge.
Scratch work: `scratchpad/assess-semantic-kernel/` (`run_page.py` runs every Python block of a page in one namespace, with the OpenAI and Azure connectors swapped for a stub chat/embedding service and a tool-calling stub; `rt/` is a .NET 10 console project that runs the .NET examples against a fake `IChatCompletionService`).

#### Scorecard

| # | Criterion | Score | Justification |
|---|---|---|---|
| 1 | Correctness of code | 3/5 | All 42 C# blocks compile and the .NET flows run. Most Python blocks run against a stub. But the promoted Python migration bridge (`as_agent_framework_tool()`) fails against every GA Agent Framework release, the Python OpenAPI example fails to load, and three Python production modules are broken. |
| 2 | Currency | 4/5 | `check-versions` is clean and every model ID is a current family (gpt-5 / gpt-5-mini / gpt-5-nano, text-embedding-3-*). Wrong current package names for Postgres/SQLite vector stores; A2A, Copilot Studio and Bedrock are labelled preview when they are alpha. |
| 3 | Accuracy of prose | 3/5 | Release history, the planner and vector-store milestones and the support policy all check out. False: Python "token usage metrics" (only .NET emits them). Root and Python diagrams contain invented behaviour (structured-output "set defaults/retry" pipeline, an A2A "message bus" with a made-up message format). |
| 4 | Coverage | 3/5 | Kernel, plugins, function calling, agents, the five orchestrations, vector stores, MCP and filters are covered well. Missing: prompt-template injection safety (`AllowDangerouslySetContent`), agent memory providers (`AIContextProvider`, Mem0/Whiteboard), realtime/multimodal, worked examples for service-backed agents (AzureAIAgent, Responses), and the .NET Process framework. |
| 5 | Structure and navigation | 3/5 | Indexes have install, a working minimal example and a page map, and links check clean. But the sidebar lists only the three index pages; root, Python and .NET production and diagram pages overlap; index cards use relative `./` links. |
| 6 | Consistency | 3/5 | The Agent Framework framing and version story are consistent. The "kernel per request" advice is contradicted by the Python production app. Revision-history tables appear on some pages only. There are American spellings in the production pages and leftover "never existed" correction rows. |
| 7 | Production readiness | 3/5 | The root production page is specific and current (retry layering, loop limits, GenAI OTel switches, transient kernel). The Python production page has broken modules and generic K8s and security filler. The .NET page is correct but thin (no security, scaling or cost content). |

**Overall verdict.** The rewrite has made this a mostly sound, current guide: the .NET code compiles and runs, and nearly all of the Python code runs against a stub. The concepts and production pages are concise and accurate. Three problems hold it back: a broken Python bridge to Agent Framework that every index page promotes, a false claim about Python token metrics that the production advice relies on, and invented content in the diagram pages. Fixing those and tidying the Python production page would bring it to 4/5.

#### Code verification results

Method. Python: each page's blocks run in order in one namespace, with stub services, plus driver code that calls the functions the page defines (tool calls exercised through a stub that emits `FunctionCallContent`). C#: `cscheck.py` with the SK csproj (`--shared`, page usings as prelude). Key flows were also run in `rt/`: minimal agent, threads, sequential and group-chat orchestration, InMemory vector store with `VectorStoreTextSearch`, the Moq test, `AsAIAgent()`, and the OpenAPI import.

| Page | Lang | Blocks | Pass | Fail | Notes |
|---|---|---|---|---|---|
| index.mdx | – | 0 | – | – | |
| semantic_kernel_comprehensive_guide.md | Py / C# | 4 / 4 | 3 / 4 | 0 / 0 | Py L239 is a fragment (no imports, `MyPydanticModel` undefined) |
| semantic_kernel_recipes.md | Py / C# | 7 / 7 | 7 / 7 | 0 | Calculator tool call verified with tool-calling stub; structured-output parse depends on a real model |
| semantic_kernel_production_guide.md | – | 0 | – | – | |
| semantic_kernel_diagrams.md | – | 0 | – | – | |
| python/index.mdx | Py | 1 | 1 | 0 | Minimal example runs |
| python/semantic_kernel_comprehensive_python.md | Py | 26 | 22 | **2** | Fail: L274 OpenAPI petstore (`Exception: Neither of the media types of uploadFile is supported.`); L706 `as_agent_framework_tool()` (ImportError with MAF 1.22.0 installed). Fragments needing undefined names: L178 (`long_text`), L438 (`triage`/`billing`/`tech`, and `ChatMessageContent`/`AuthorRole` not imported). MCP L578 verified with a local FastMCP stdio server; L601 `as_mcp_server` signature and `Server` object verified. |
| python/semantic_kernel_recipes_python.md | Py | 8 | 8 | 0 | Needs `tenacity` (not an SK dependency). Filesystem MCP pattern verified through the local-server equivalent. |
| python/semantic_kernel_advanced_multi_agent_python.md | Py | 5 | 5 | 0 | Review loop, routing and typed `ConcurrentOrchestration[str, Opinions]` run |
| python/semantic_kernel_middleware_python.md | Py | 4 | 4 | 0 | Redaction, timing, approval gate (tool call blocked, model sees message) all behave as described |
| python/semantic_kernel_production_python.md | Py | 8 | 5 | **3** | main.py, monitoring.py and the instrumented block fail (see Blockers); pytest stub test passes |
| python/semantic_kernel_streaming_server_fastapi.md | Py | 1 | 1 | 0 | SSE endpoints return `event: token`/`done` via TestClient; thread stored |
| dotnet/index.mdx | C# | 1 | 1 | 0 | Runs end to end |
| dotnet/semantic_kernel_comprehensive_dotnet.md | C# | 21 | 21 | 0 | Compiles; threads, sequential, vector store and `AsAIAgent()` run. Petstore import works in .NET (15 functions) |
| dotnet/semantic_kernel_recipes_dotnet.md | C# | 5 | 5 | 0 | `WeatherPlugin` comes from the comprehensive page (stated) |
| dotnet/semantic_kernel_production_dotnet.md | C# | 4 | 4 | 0 | Moq test body runs and passes |
| diagrams (py, dotnet) | – | 0 | – | – | |
| **Total** | | **Py 64 / C# 42** | **Py 56 / C# 42** | **Py 5 / C# 0** | Py: 3 fragments depend on names defined elsewhere |

Compiling without the shared-usings prelude shows standalone snippets missing `using System.ComponentModel;` (root recipes L133, root concepts L122) and `Microsoft.SemanticKernel.ChatCompletion` / DI usings (dotnet production L131).

Tooling: `node scripts/check-versions.mjs | grep semantic` is clean. `node scripts/check-links.mjs` reports 0 broken links across the site.

#### Issues

### Blocker

1. **Python Agent Framework bridge is broken against current MAF.** In SK 1.45.0, `KernelFunction.as_agent_framework_tool()` does `from agent_framework import AIFunction`. That class is called `FunctionTool` in every GA release (checked 1.0.0 and 1.22.0), so the call raises "agent_framework is not installed" even when it is installed. The pages present it as the migration path: python/index.mdx:30, :84; python/semantic_kernel_comprehensive_python.md:704-708; index.mdx:27; semantic_kernel_comprehensive_guide.md:36. *Fix:* state that the bridge only works with pre-GA `agent-framework` builds (pin, or wait for an SK fix). Offer the working alternative instead: wrap the SK function in a plain Python function or `@tool`, or `FunctionTool(...)` calling `kernel.invoke`. Also say that `agent-framework-core` must be installed.
2. **Python OpenAPI example fails as written.** python/semantic_kernel_comprehensive_python.md:277-281: `add_plugin_from_openapi(... petstore3 openapi.json ...)` raises `Exception: Neither of the media types of uploadFile is supported.` in 1.45.0 (the same document imports fine in .NET). *Fix:* use a spec SK's Python parser accepts, or pass `operations_to_exclude=["uploadFile"]` in `OpenAPIFunctionExecutionParameters`, and re-run.
3. **False claim: Python emits token-usage metrics.** Python SK 1.45.0 defines only `semantic_kernel.function.invocation.duration` and `.streaming.duration` histograms. `...token_usage.prompt/completion` exist only in .NET (`KernelFunctionFromPrompt.cs`). Affected: semantic_kernel_comprehensive_guide.md:253 ("Both SDKs ... token usage per function"), semantic_kernel_production_guide.md:67 and :111 ("Track token usage ... from SK's metrics"), python/semantic_kernel_production_python.md:346 and :412-413. *Fix:* say token-usage metrics are .NET-only. In Python, read usage from `ChatMessageContent.metadata["usage"]` (as recipes_python:236 already does) or from GenAI span attributes, and record it with the custom `token_counter`.
4. **Python production modules do not run.** In python/semantic_kernel_production_python.md, `monitoring.py` raises `NameError: Config` at import (:353, no import of `Config`). The instrumented block uses `latency_histogram`, which is local to `setup_monitoring` (:410). `main.py` uses undefined `ChatRequest` (:271) and `logger` (:279) and imports a `health.HealthCheck` module that the page never defines (:238, :249). *Fix:* add `from config import Config`, return or hold the instruments at module level, define `ChatRequest(BaseModel)` and `logger`, and either show `health.py` or drop it.
5. **Invented A2A architecture.** python/semantic_kernel_diagrams_python.md:135-161 shows SK agents on an "A2A Message Bus" with a "Message Router" talking to LangGraph and AutoGen agents, with a made-up message format (`type: REQUEST`, `sender`, `correlation_id`). The comprehensive page (:618-620) correctly says Python SK has no A2A connector. *Fix:* delete the diagram, or replace it with "SK agent wrapped by `a2a-sdk` server ⇄ A2A JSON-RPC client".

### Major

6. **Wrong current package names for Postgres/SQLite vector stores.** `Microsoft.SemanticKernel.Connectors.PgVector` / `.SqliteVec` are themselves deprecated (last 1.74.0-preview; the NuGet deprecation message reads "renamed ... to CommunityToolkit.VectorData.PgVector / .SqliteVec"). Affected: dotnet/index.mdx:76, dotnet/semantic_kernel_comprehensive_dotnet.md:365, semantic_kernel_comprehensive_guide.md:206. *Fix:* use `CommunityToolkit.VectorData.PgVector` and `CommunityToolkit.VectorData.SqliteVec`, and add `.AzureCosmosDB`/`.CosmosMongoDB` to the list.
7. **Root diagrams describe behaviour SK does not have.** In semantic_kernel_diagrams.md:
   - :31-35 lists "Semantic" plugins and "Volatile" memory.
   - :94-96 has an "Update Memory / update embeddings" step after every request.
   - :202-320 shows generic Master-Worker, Peer-to-Peer and Hierarchical patterns instead of SK's five orchestrations.
   - :461 uses `"Math:Add"` (SK tool names are `Math-Add`).
   - :493 shows "Tokens: 5" on a native call.
   - :507-580 shows a structured-output "validation / set defaults / retry" pipeline that SK does not run.
   - :679-683 gives "Max iterations: 10 / Error after 3 retries" loop control.
   - :730 ends with a fluff closing sentence.

   *Fix:* replace sections 1.1, 1.2, 2.2, 3, 6, 7 and 9 with diagrams of the real kernel, filter pipeline and the sequential/concurrent/group/hand-off/Magentic orchestrations (the dotnet diagrams page, :59-81, already does this well). Drop the closing sentence.
8. **Contradictory kernel-lifetime advice.** semantic_kernel_production_guide.md:31 and the comprehensive pages say to build the kernel per request. python/semantic_kernel_production_python.md:245-248 builds one kernel at start-up and shares it, and the streaming server (python/semantic_kernel_streaming_server_fastapi.md:26-32) does the same. *Fix:* keep the service in `app.state` and build a `Kernel` per request in a FastAPI dependency. If the singleton is deliberate, say why.
9. **Sidebar exposes only the three index pages.** astro.config.mjs:310-316 has no entries for the concepts, recipes, production and diagram pages or the Python/.NET subpages. Readers can reach them only through index cards. *Fix:* add nested Python/.NET groups (comprehensive, recipes, advanced, middleware, streaming, production, diagrams) as the Agent Framework entry does.
10. **Missing security feature: prompt-template input encoding.** No page mentions that SK HTML-encodes inserted variables and function results by default, or the `AllowDangerouslySetContent` / `allow_dangerously_set_content` opt-outs. This is a core part of SK's prompt-injection story. *Fix:* add a short section to the comprehensive pages (prompt functions) and a bullet under Security in semantic_kernel_production_guide.md:78.
11. **Coverage gaps for features SK ships today.** These are absent:
    - agent memory providers (`AIContextProvider`, `Mem0Provider`, `WhiteboardProvider` in .NET);
    - realtime (`semantic-kernel[realtime]`) and image/audio services;
    - worked examples for `AzureAIAgent` / `OpenAIResponsesAgent` (tables only: python comprehensive:377-385, dotnet comprehensive:305-311);
    - the .NET Process framework (one table row, dotnet comprehensive:52);
    - .NET chat-history reducers.

    *Fix:* add one short verified example each for a Responses agent and an Azure AI agent, a paragraph and snippet on context providers, and mention realtime in passing.
12. **Package status labels wrong.** `Microsoft.SemanticKernel.Agents.A2A` is `1.81.0-alpha`, not preview: dotnet/semantic_kernel_comprehensive_dotnet.md:48, :311, :445 and semantic_kernel_comprehensive_guide.md:217. Copilot Studio and Bedrock agent packages are `-alpha`, not "preview packages": dotnet comprehensive:310. *Fix:* correct the suffixes.
13. **Python production page is mostly generic filler.**
    - :22-33 is a generic checklist that duplicates the root page's release checklist.
    - :105-226 is a generic K8s Deployment/HPA with nothing SK-specific.
    - :587-596 is an eight-word "Security Best Practices" list.
    - :47-49 installs `build-essential` for no reason.
    - :93-96 passes `OPENAI_API_KEY` while the code (:287) uses Azure with managed identity.

    *Fix:* cut the K8s YAML to the probes, cut the generic checklist, link to the root page for security, and make the env vars match the code.

### Minor

14. Index cards use relative `./...` links, against AGENTS.md's root-absolute rule: python/index.mdx:36-78, dotnet/index.mdx:36-71, index.mdx:50-65. *Fix:* use `/semantic-kernel-guide/python/...`.
15. Fragments with undefined names: python comprehensive:187 (`long_text`), :441-451 (`triage`/`billing`/`tech` undefined; `ChatMessageContent`, `AuthorRole` not imported); semantic_kernel_comprehensive_guide.md:239-241 (no imports). *Fix:* define the names or point to where they are defined, and add the imports.
16. Prompt-directory wording, python comprehensive:223-226. `parent_directory="./prompts", plugin_name="WriterPlugin"` loads `./prompts/WriterPlugin/<function>/skprompt.txt`, but the text says "a plugin directory ... loads with". *Fix:* show the folder tree.
17. Unstated extra dependencies: `tenacity` (python recipes:212; production:429), `circuitbreaker`, and `pytest-asyncio` for the test (production:576). *Fix:* add a `pip install` line.
18. Root recipes, "All C# snippets assume" (semantic_kernel_recipes.md:22-29), omits `using System.ComponentModel;` (needed by `Description` at :135-139). The same applies to semantic_kernel_comprehensive_guide.md:121-126. dotnet production test (:130-132) lacks the `Microsoft.SemanticKernel.ChatCompletion` and DI usings. *Fix:* add the usings.
19. Recipe 7 asks a compound-interest question but `Calculator` (semantic_kernel_recipes.md:118-126) has only `add` and `power`, so the model must multiply unaided. That undercuts the point made at :147. *Fix:* add `multiply`.
20. dotnet comprehensive:359 says "Hand-off and group chat accept an `InteractiveCallback`". For group chat it is on the manager (`GroupChatManager.InteractiveCallback`). *Fix:* say so.
21. python advanced:47 says "Subclass `GroupChatManager`" but the code subclasses `RoundRobinGroupChatManager`. `GroupChatManager`, `StringResult` and `MessageResult` are imported unused (:50). *Fix:* align the wording and trim the imports.
22. American spellings: "Optimization" (python production:16, :479; dotnet production:16), "Containerize/Containerization" (python production:24, :37; dotnet production:22), "analyzer/Analyze" (python diagrams:142, :158-159). *Fix:* use British spelling in prose.
23. Leftover correction framing in reader-facing tables and text: "Never existed" rows (python comprehensive:723-724; dotnet comprehensive:586-587), and "earlier versions of this page/recipe used..." (python advanced:10; python recipes:171, :267; dotnet recipes:165). *Fix:* keep this history in the revision tables only.
24. Bold taglines that repeat the title: python production:8, dotnet production:8, python diagrams:8. "MCP Architecture (2025)" heading: python diagrams:99. *Fix:* remove them.
25. Revision-history tables are inconsistent: present on the indexes, concepts, recipes, root production and comprehensive pages, absent on python recipes, advanced, middleware, streaming, production and diagrams and on the .NET recipes, production and diagrams pages. *Fix:* add them everywhere or nowhere.
26. python diagrams:52-58 draws `await kernel.invoke(...)` lines feeding `asyncio.gather()`, which suggests sequential awaits. *Fix:* show `gather(kernel.invoke(f1), kernel.invoke(f2), ...)`.
27. semantic_kernel_production_guide.md:79 says to "restrict with function-choice filters". *Fix:* name the mechanism: `FunctionChoiceBehavior.Auto(filters={"included_plugins": [...]})` / `Auto(functions: [...])`.

#### Top 5 improvements

1. Fix or replace the Python Agent Framework bridge (Blocker 1) on all five pages that promote it, and add a verified MAF-side snippet.
2. Correct the Python observability story: token usage is not a Python metric (Blocker 3). Then repair the Python production modules so they import and run (Blocker 4), and align them with the kernel-per-request advice (Major 8).
3. Rewrite the root and Python diagram pages around SK's real mechanisms (filter pipeline, the five orchestrations, vector store flow). Delete the invented A2A, structured-output and generic multi-agent diagrams (Blocker 5, Major 7).
4. Close the coverage gaps that matter in production: prompt-template encoding and `AllowDangerouslySetContent`, agent memory providers, and one worked Responses/Azure AI agent example per language (Majors 10-11).
5. Fix navigation and package facts: nested sidebar entries for every page, root-absolute links, and the correct `CommunityToolkit.VectorData.PgVector/SqliteVec` names and alpha/preview labels (Majors 6, 9, 12; Minor 14).

---

### Haystack guide assessment (`src/content/docs/haystack-guide/`)

Assessed 2026-10-21 against `haystack-ai` 3.4.1 (versions.json; PyPI latest, released 2026-10-21), plus anthropic-haystack 8.0.0, elasticsearch-haystack 7.1.0, qdrant-haystack 13.0.0, mcp-haystack 2.1.0, serperdev-haystack 2.0.0, opentelemetry-haystack 2.0.0 and langfuse-haystack 8.0.0. Venv: `scratchpad/haystack/venv`. Upstream source at tag v3.4.1: `scratchpad/haystack/src`. Scratch scripts: `scratchpad/assess-haystack/` (`run_page.py`, `introspect.py`, `behave.py`, `statefix.py`, `fastapi_test.py`, `misc.py`).

Pages read in this order: index.mdx, start_here.md, haystack_comprehensive_guide.md, haystack_advanced_agents_python.md, haystack_multi_agent_guide.md, haystack_recipes.md, haystack_observability_python.md, haystack_production_guide.md, haystack_streaming_server_fastapi.md, haystack_diagrams.md. There is also a stray `rag.yaml` in the folder (see B1).

#### Scorecard

| # | Criterion | Score | Justification |
|---|-----------|:-----:|---------------|
| 1 | Correctness of code | 4 | 70 of the 76 Python blocks run cleanly, and the other 6 fail only because of harness artefacts or deliberate fragments. Every block imports and runs against 3.4.1 with a signature-checking mock. Both FastAPI services pass end to end. Every keyword argument I checked matches the real signature. However, two examples do not do what the prose says: tenant isolation, and late-set content tracing. |
| 2 | Currency | 3 | `check-versions` is clean and the 3.x API story is current. But the "Python 3.10+" claim is wrong (3.4 needs ≥3.11), `claude-sonnet-4-5` appears twice, and newer 3.1–3.4 features are missing (built-in token budget, compaction, offloading, SkillToolset). |
| 3 | Accuracy of prose | 3 | Most behaviour claims check out against source: defaults, exit reasons, span names, allowlist, warm-up key resolution, breakpoints. One false security claim is the headline of a recipe, plus an incomplete reserved-keys list and a few mismatched cross-references. |
| 4 | Coverage | 3 | Good on Agent, tools, state, hooks, HITL, pipelines, routing and serialisation. Missing entirely: evaluation, embedding/indexing in depth, multimodal, built-in context-management hooks, SkillToolset, memory stores, Hayhooks and telemetry. Multi-agent material is covered three times. |
| 5 | Structure & navigation | 3 | The index has install, a minimal example and a page map, and links pass. But the sidebar exposes only the index, nine pages lack `language`, a stray YAML test artefact sits in the content folder, and the diagrams page has a TOC gap and generic, invented diagrams. |
| 6 | Consistency | 4 | One API style (`Agent(chat_generator=…)`, `run(messages=…)`), one version story, placeholders everywhere, understated tone. A few lapses: American "Specialization", relative index links, a fluff closing line in the diagrams page. |
| 7 | Production readiness | 3 | The checklist, allowlisted YAML, warm-up/close, the FastAPI service, SSE proxy notes, caching and CacheChecker are real and current. Security, testing/evaluation, cost controls and telemetry opt-out are explicitly "not covered", and the security advice that is given (state-injected tenant IDs) is unsafe. |

**Verdict.** This is a solid, mostly accurate Haystack 3 guide: the code really runs on 3.4.1, and the 2.x→3.x migration story is careful and well sourced. Two things hold it back. A false security claim in the multi-tenant recipe (also repeated as general advice) must be fixed before anyone copies it. And the guide stops at roughly 3.0-era features: the built-in budget/compaction/offload hooks, SkillToolset and evaluation from 3.1–3.4 are absent, and production guidance skips security, testing and cost.

#### Code verification

Method: `run_page.py` pulls every fenced `python` block from a page and runs them in order in one shared namespace (cwd = scratch dir). `OpenAIChatGenerator` and `AnthropicChatGenerator` are swapped for a `MockChatGenerator` subclass that first binds the given kwargs against the **real** constructor signature, so a wrong kwarg would fail. Separately:
- `behave.py` uses a scripted `MockChatGenerator(response_fn=…)` that issues real `ToolCall`s. It checks exit-condition tools, `on_exit`+`continue_run`, `stop_run`, state merge, `inputs_from_state`, `AgentTool` + `outputs_to_state`, and streaming chunk shapes.
- `fastapi_test.py` runs the production `/health` and `/query` endpoints and the SSE `/stream` endpoint with `TestClient` against a tool-calling mock.
- `misc.py` covers the YAML round-trip with the real generator, `Pipeline.stream` handle, `PipelineRuntimeError.component_name`, `CacheChecker`, `BranchJoiner` output name, leaf-only outputs, and API-key resolution at warm-up.

YAML blocks were parsed with `yaml.safe_load_all`. Integration package names were checked on PyPI; all 13 exist.

| Page | Python blocks | Pass | Fail | Notes |
|------|:---:|:---:|:---:|-------|
| index.mdx | 1 | 1 | 0 | Minimal example also exercised end to end with a tool-calling mock (via the same agent shape in fastapi_test). |
| start_here.md | 0 | – | – | |
| haystack_comprehensive_guide.md | 39 | 35 | 3 (0 real) | L428 fails only because the mock lives in `__main__` (allowlist working as documented). The real-generator round-trip passes in misc.py. L441 is a fragment needing `agent.yaml`. L856 fails only because the echo mock never calls `final_answer`; with a scripted tool call, `last_message.tool_call_result.result` is correct. L1255 (the 2.x "before" block) was intentionally skipped. |
| haystack_advanced_agents_python.md | 5 | 4 | 1 (0 real) | L108: the echo mock returns non-JSON. `response_format` accepting a Pydantic model is confirmed in `openai.py:172`. |
| haystack_multi_agent_guide.md | 11 | 11 | 0 | But L419–423 silently fails to enable content tracing (M2). |
| haystack_recipes.md | 8 | 8 | 0 | But the L173 security claim is false (B2). |
| haystack_observability_python.md | 5 | 5 | 0 | |
| haystack_production_guide.md | 7 | 5 | 2 (0 real) | L75 and L84 are deliberate fragments (`build_rag_pipeline()` placeholder; reads the file L75 writes). The FastAPI app passes end to end: `/health` 200, `/query` 200 with `exit_reason`, `steps`, `tool_calls` and `total_tokens` populated. |
| haystack_streaming_server_fastapi.md | 1 | 1 | 0 | End to end: SSE emits `tool_call` → `tool_result` → `token`… → `done`, exactly as described. |
| haystack_diagrams.md | 0 | – | – | ASCII only. |
| **Total** | **77 Python blocks (76 run, 1 deliberate 2.x block skipped)** | **70** | **6, none real: 3 harness artefacts and 3 deliberate fragments** | All 3 YAML blocks parse, and all 13 pip package names exist on PyPI. |

Behaviour checks that passed:
- `max_agent_steps=100`, `tool_concurrency_limit=4`, `required_variables="*"`.
- Exit reasons are text/length/content_filter/max_agent_steps/tool name/`stop_run` value.
- `stop_run` from `before_llm` gives exit_reason `token_budget`.
- `on_exit` + `continue_run` loops back as described.
- List state keys append, and `AgentTool(outputs_to_state=…)` copies sub-agent facts.
- `LLMMessagesRouter` has an `unmatched` output, and the `BranchJoiner` output is `value`.
- Missing key: construction succeeds and `warm_up()` raises `ValueError`.
- `PipelineRuntimeError.component_name` is set, and `Pipeline.run` returns leaf outputs only.
- `stream()` handle `.result` and `chunk.component_info.name` work.
- Span names match `agent.py:747/1068/1088` and `tool_calling.py:197`.
- `HAYSTACK_AUTO_TRACE_ENABLED` has no effect (MIGRATION.md:183).

#### Issues

### Blocker

| ID | File:line | Problem | Suggested fix |
|----|-----------|---------|---------------|
| B1 | `haystack-guide/rag.yaml` (whole file) | A stray test artefact sits in the content folder. It is the serialised RAG pipeline from a harness run, containing `type: __main__.FakeCls` (added in commit 5ad4b28 "WIP"). It is not referenced as a page and would fail to load for anyone who tried it. | Delete the file. Run future verification from a scratch directory, not the docs folder. |
| B2 | haystack_recipes.md:173 (also :105 "the model never sees or chooses it") | False security claim: "Because `tenant_id` is not part of any tool's JSON schema, a prompt-injected request cannot switch tenants." In 3.4.1, **LLM-supplied arguments take precedence over `inputs_from_state`** (`tool_calling.py:421`: "LLM args take precedence and short-circuit the state lookup"). Reproduced: a tool call with `{"customer_id":"C-1","tenant_id":"globex"}` in an `acme` run returned `globex:C-1` (behave.py). | Read the tenant from a `State`-typed parameter instead (`def search_kb(query: str, state: State): tenant_id = state.get("tenant_id")`). It is injected by the Agent and cannot be overridden: verified in statefix.py, where an injected `tenant_id` makes the call fail rather than switch tenant. Then reword L173 accordingly. |
| B3 | haystack_advanced_agents_python.md:177 | Same unsafe advice as general guidance: "Values such as tenant or user IDs belong in `State` via `inputs_from_state`, not in the tool schema." | Say that `inputs_from_state` is a convenience, not a security boundary (the model can override it), and recommend a `State` parameter or `hook_context` for privileged values. |
| B4 | index.mdx:15, index.mdx:24, haystack_comprehensive_guide.md:53 | "Python 3.10+" / "requires Python 3.10 or later" is false. `pyproject.toml:11` has `requires-python = ">=3.11"`, and the release note says "Dropped support for Python 3.10". `pip install haystack-ai==3.4.1` fails on 3.10. | Change to "Python 3.11+". Add one line to the migration table: "Python 3.10 support dropped". |

### Major

| ID | File:line | Problem | Suggested fix |
|----|-----------|---------|---------------|
| M1 | haystack_comprehensive_guide.md:338, :955 | Stale model ID `claude-sonnet-4-5` (it is the anthropic-haystack default, but the Claude 5 family is current). | Use `claude-sonnet-5-5`. |
| M2 | haystack_multi_agent_guide.md:419–423 | `os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")` runs after `haystack` has been imported. `ProxyTracer.__init__` reads the variable once at import (`tracer.py:118`), so content tracing stays **off**; verified, it prints `False`. observability:63 correctly says "set before importing haystack", so the pages contradict each other. | Replace with `from haystack import tracing; tracing.tracer.is_content_tracing_enabled = True`, or move the env var to the top of the script or deployment. |
| M3 | astro.config.mjs:195 | The sidebar has a single "Haystack" link, so nine guide pages are reachable only from index cards. Other multi-page guides (e.g. MAF, astro.config.mjs:221–235) list every page. | Make Haystack a collapsed group: Overview, Start here, Comprehensive guide, Advanced agents, Multi-agent, Recipes, Observability, Production, Streaming server, Diagrams. |
| M4 | haystack_advanced_agents_python.md:28–38; haystack_comprehensive_guide.md:796 | The guide hand-rolls a token-budget hook and suggests summarising history with "a second, cheap model". Haystack ships these features (experimental): `haystack.hooks.budget.TokenBudgetHook` (exit_reason `token_budget_exceeded`), `haystack.hooks.compaction.CompactionHook` with `SlidingWindowCompactor` / summarisation / tool-result pruning compactors, and `haystack.hooks.tool_result_offloading.ToolResultOffloadHook`. | Keep the custom `TokenBudget` as an example of the Hook protocol, but lead with `TokenBudgetHook`. Add a "Context management" section on `CompactionHook` and `ToolResultOffloadHook` (marked experimental). Point L796 to it. |
| M5 | (missing) haystack_comprehensive_guide.md:545–557 tool table | `SkillToolset` (README "Features"; `haystack/tools/skills/skill_toolset.py`) is absent from the tool-type table and the whole guide. | Add a row and a short example next to `SearchableToolset`. |
| M6 | (missing) whole guide; comprehensive:63 lists "evaluators" | Evaluation is never covered, although the core ships evaluators and upstream docs have `optimization/evaluation`. There is no testing guidance beyond one sentence (production:65). | Add an "Evaluating and testing" section: `MockChatGenerator` in pytest, `FaithfulnessEvaluator` / `ContextRelevanceEvaluator` / `DocumentMRREvaluator` on a small golden set, and running evaluators in CI. |
| M7 | haystack_production_guide.md:422 | "Security hardening, rate limiting, … are not covered." The production page has no security section. It also never mentions that Haystack sends anonymous telemetry by default (`haystack/telemetry/_telemetry.py:26`, README "Telemetry"). | Add a short security section: `Pipeline.load` allowlist, `State`/`hook_context` for privileged IDs (see B2), `ConfirmationHook` for side effects, content-tracing PII, `max_agent_steps`/`TokenBudgetHook` for cost. Add a line "set `HAYSTACK_TELEMETRY_ENABLED=False` to opt out". |
| M8 | haystack_comprehensive_guide.md:852 | The reserved `state_schema` keys list is incomplete. The source reserves `step_count, token_usage, tool_call_counts, exit_reason, continue_run, stop_run, tools, hook_context, context_tokens` (`agent.py:88–111`). | List all nine names, or link to MIGRATION.md "Reserved `state_schema` keys". |
| M9 | haystack_diagrams.md:330–373, :557–589, :776–804, :808–864, :866–893 | Generic or invented diagrams. They show an IDLE/PROCESSING/TOOL_EXECUTION state machine Haystack does not have, and a "Document Abstraction Layer / Index Management / Embedding Management" that does not exist. There is also a Kafka/RabbitMQ microservices picture and a generic ReAct "REFLECT" loop unrelated to Haystack APIs. | Delete these five. Keep the accurate ones (Agent Execution Flow :275–328, observability spans :671–719, pipeline flows). Add a SuperComponent/PipelineTool diagram and a hooks timeline instead. |
| M10 | Duplication: comprehensive:1097–1179, multi_agent_guide.md:72–211, recipes:273–327; and resilience lists at comprehensive:936–942, observability:102–106, production:350–354 | Coordinator/AgentTool, sequential stage and fan-out patterns are written out three times, and the retry/fallback bullet list three times. | Keep a short pointer in the comprehensive guide (one example each), the full treatment in the multi-agent guide, and the recipe as an end-to-end app only. Keep resilience in one place (observability) and link to it. |
| M11 | haystack_comprehensive_guide.md:1230–1251 | The migration table omits two upstream breaking changes from MIGRATION.md: the `GeneratedAnswer`/`ExtractedAnswer` serialisation format (MIGRATION.md:9) and "Confirmation strategies now see only the model-requested tool arguments" (MIGRATION.md:675). It also omits the Python 3.10 drop (B4). | Add three rows. |

### Minor

| ID | File:line | Problem | Suggested fix |
|----|-----------|---------|---------------|
| m1 | advanced_agents:4, comprehensive:4, diagrams:4, multi_agent:4, observability:4, production:4, recipes:4, streaming:4, start_here:4 | Frontmatter lacks `language: python` (only index.mdx has it). | Add `language: python` after `framework: haystack`. |
| m2 | All pages (44 occurrences, e.g. index.mdx:106, comprehensive:254) | `gpt-5-mini` is the library default and still in `SUPPORTED_MODELS`, but upstream release notes now use `gpt-5.4-nano` / `gpt-5.4-mini`, and `gpt-5.5` exists. | Optional: move examples to `gpt-5.4-mini` and keep the "model names change" note (comprehensive:348). |
| m3 | index.mdx:32 | "Read **Migrating from Haystack 2.x**" is bold text, not a link. | Link to `/haystack-guide/haystack_comprehensive_guide/#migrating-from-haystack-2x`. |
| m4 | index.mdx:38–78, :115 | Internal links are relative (`./haystack_comprehensive_guide/…`); AGENTS.md asks for root-absolute links. | Use `/haystack-guide/haystack_comprehensive_guide/…`. |
| m5 | haystack_multi_agent_guide.md:186 | Says the pipeline loop uses "a `ConditionalRouter` and a `BranchJoiner` (see Iterative refinement loop)". The linked example uses a custom `LengthValidator` with two outputs, not a `ConditionalRouter`. | Reword: "with a validator component and a `BranchJoiner`". |
| m6 | haystack_advanced_agents_python.md:134–168 | The text says to release clients in `close()`, but the example implements only `close_async()`, and sync `run()` ignores the client entirely (`httpx.get`). | Add a sync `close()` (or say async-only components need only `close_async`), and either use an `httpx.Client` in `run` or say sync calls are one-shot. |
| m7 | haystack_multi_agent_guide.md:15, :215 | American spelling "Specialization". | "Specialisation" (the anchor changes; update L15). |
| m8 | haystack_diagrams.md:9–20, :897 | The TOC omits "Advanced Pattern Diagrams". The closing line "comprehensive visual representations … across all levels of complexity" is fluff. | Fix the TOC (or drop the section per M9) and delete L897. |
| m9 | haystack_recipes.md:189 | The recipe writes `report.txt` into the cwd; together with comprehensive:432 (`rag.yaml`), this is how B1 happened. | Use `tempfile.TemporaryDirectory()` in examples that write files. |
| m10 | haystack_production_guide.md:342 | Hayhooks is mentioned in one vague sentence ("consult its documentation"). | Add a minimal `hayhooks run` / pipeline-wrapper example, or at least the command and what it exposes (REST + OpenAI-compatible + MCP). |
| m11 | src/data/frameworks.ts:151 (outside the folder) | The home-card tagline mentions "Deepset Studio", which the guide never covers; the brand is "deepset". | "Pipelines and agents for production RAG and search." |
| m12 | haystack_comprehensive_guide.md:1083–1094 | `max_runs_per_component=3` makes the loop raise when validation keeps failing (no graceful exit), and nothing says so. | Add one sentence: the pipeline raises once a component exceeds `max_runs_per_component`, so catch it or add a counter route. |

#### Top 5 improvements

1. **Fix the tenant-isolation example and advice** (B2, B3). Use a `State`-typed parameter, and say plainly that LLM arguments override `inputs_from_state`. This is the only thing in the guide that could cause real harm.
2. **Correct the version facts and clean up the folder.** Python 3.11+ (B4, M11), delete `rag.yaml` (B1), use `claude-sonnet-5-5` (M1), and fix the late content-tracing env var (M2).
3. **Add the 3.1–3.4 agent features.** `TokenBudgetHook`, `CompactionHook` with its compactors, `ToolResultOffloadHook` and `SkillToolset` (M4, M5), each marked experimental where upstream says so.
4. **Make the production page production-grade.** Add sections on security, testing and evaluation (`MockChatGenerator` in CI plus core evaluators), cost controls and the telemetry opt-out (M6, M7), and a concrete Hayhooks example (m10).
5. **Tighten navigation and remove padding.** Turn the sidebar entry into a full group (M3), add `language` frontmatter (m1), cut the invented diagrams (M9), and de-duplicate the multi-agent and resilience material so each topic has one home (M10).

---

### Assessment: Anthropic Claude Agent SDK guides (Python and TypeScript)

Scope: `src/content/docs/anthropic-claude-agent-sdk-guide/` (Python, 6 pages) and `src/content/docs/anthropic-claude-agent-sdk-typescript-guide/` (TypeScript, 9 pages). Assessed 2026-10-21 against `claude-agent-sdk` 0.2.170 (PyPI latest, bundled CLI 2.1.306) and `@anthropic-ai/claude-agent-sdk` 0.3.306 (npm latest). Both match `src/data/versions.json`.

Scratch work: `/tmp/claude-0/-home-user-AgentGuides/55b0d51f-358a-5617-8094-250bd4df3d90/scratchpad/assess-claude-agent-sdk/` (extracted blocks, a fake stream-json Claude Code CLI `tools/fake_claude` used as the mock model, runners, upstream CHANGELOGs).

#### Scorecard

| # | Criterion | Score | Justification |
|---|---|---|---|
| 1 | Correctness of code | 3 | Every Python block parses, every SDK import and dataclass keyword resolves, and every TS block passes `tsc --strict`. All the runnable Python examples run end to end against the fake CLI. But the error-handling helpers fail at runtime: the Python retry helpers never retry, and the TS retry helper retries the limit errors it says it leaves alone. |
| 2 | Currency | 4 | `check-versions` is clean for both folders. Model IDs are all Claude 5 family and nothing stale was found. Only the Python pages still list `TodoWrite` as a current built-in tool. |
| 3 | Accuracy of prose | 3 | The release history matches both CHANGELOGs line for line. There are still several false claims: the `setting_sources` default, hooks "never firing", `load_timeout_ms`, the TS "errors arrive as a result message" claim, and invented compaction details in the diagrams. |
| 4 | Coverage | 4 | The core API is covered well (query/client, tools, MCP, permissions, hooks, subagents, sessions, structured output, thinking, checkpointing). Gaps: user-input/AskUserQuestion, SessionStore implementations, image input, skills/plugins depth, and Python has no streaming-server or middleware page. |
| 5 | Structure and navigation | 4 | Each index gives install, a minimal example and a page map. 0 broken links, no stubs, no deep-dive pages, no H1s. The sidebar shows only "Overview" per language, index links are relative, and the diagrams pages are partly filler. |
| 6 | Consistency | 3 | Three different `framework` values, `language` missing on 13 of 15 pages, uneven titles and file names, and the two languages tell different stories about `setting_sources`. |
| 7 | Production readiness | 3 | Concrete and current: budgets, `model_usage` cost accounting, OTel env, hardening options, Docker, K8s notes, troubleshooting. Both languages' headline resilience helpers are broken (see Blockers), and the TS Dockerfiles have no build step. |

**Verdict:** These are recent, careful rewrites. The API surface is accurate, every block compiles, and the version history is correct. The weak spot is failure semantics: both SDKs yield an error `result` message and then raise or throw, and the guides' retry and fallback code does not account for that. Retries silently never happen (Python) or happen when they shouldn't (TS). Fix the six blockers and the frontmatter, and this is a strong guide pair.

#### Code verification results

Method:
- **Python.** Every ```` ```python ```` block was extracted and `ast.parse`d. Each `from claude_agent_sdk import …` name was checked against the installed package, and every keyword passed to SDK dataclasses (`ClaudeAgentOptions`, `AgentDefinition`, `HookMatcher`, `PermissionResult*`, `ToolPermissionContext`) was checked against the real fields. Every block was then executed with `cli_path` pointed at a fake Claude Code CLI that speaks the stream-json control protocol (init, assistant, result, structured output, partial events, and `get_context_usage`/`mcp_status` control replies). Missing `cwd` paths (`/srv/...`) were swapped for a temp dir. Recipe 14 was run under pytest. Dict-typed options (`sandbox`, `thinking`, `task_budget`, preset `exclude_dynamic_sections`, hook outputs, `get_mcp_status`/`get_context_usage` keys, `ModelUsage` keys) were checked by hand against the TypedDicts.
- **TypeScript.** Every ```` ```typescript ```` block went into its own module, with `run.ts` and `policies.ts` stubbed from the guide's own helpers. All were compiled with `tsc --strict`, NodeNext, TS 5.9.3, Zod 4.6.7 and Express types. The index example ran end to end with `pathToClaudeCodeExecutable` set to the fake CLI.
- **Error paths.** The fake CLI was switched to emit an error result (`error_during_execution` with `api_error_status: 529`, or `error_max_turns`) and exit 1, the way the real CLI does. The retry helpers were then run against it.

| Page | Blocks | Static pass | Ran OK | Failed (behaviour) | Notes |
|---|---|---|---|---|---|
| py `index.mdx` | 2 | 2 | 2 | 0 | Both print `42` |
| py comprehensive | 28 | 28 | 24 | 1 | 4 non-runnable by design: L104 needs `ANTHROPIC_API_KEY`, L726 uses placeholder `SESSION_ID`, L887 and L900 are signature stubs. **L851 `run_with_retry` fails its behaviour test** |
| py recipes | 15 | 15 | 15 | 0 | Recipe 14 under pytest: 2 passed, 1 skipped (needs API key). Recipe 8 emits `CanUseToolShadowedWarning`, which is intended |
| py production | 9 | 9 | 9 | 1 | **L39 `run_resilient` fails its behaviour test** |
| py migration | 6 | 5 | 2 | 0 | L47 imports `claude_code_sdk` on purpose ("Before"); L72/L77/L94 are fragments without imports |
| **Python total** | **60** | **59** | **52** | **2** | |
| ts comprehensive | 22 | 22 | – | 0 | |
| ts production | 10 | 10 | – | 1 | **L25 `runResilient` fails its behaviour test** |
| ts recipes | 8 | 8 | – | 0 | |
| ts middleware | 6 | 6 | – | 0 | |
| ts migration | 5 | 5 | – | 0 | |
| ts advanced multi-agent | 3 | 3 | – | 0 | L18 router fallback branch is unreachable (see B4) |
| ts express | 2 | 2 | – | 0 | |
| ts `index.mdx` | 1 | 1 | 1 | 0 | Prints `42` |
| **TypeScript total** | **57** | **57** | 1 run | **1** | |

Also checked:
- `node scripts/check-versions.mjs`: no hits for either folder.
- `node scripts/check-links.mjs`: 0 broken links across the site.

#### Issues

### Blocker

**B1. Python comprehensive guide: the retry helper never retries and returns the error text as the answer.**
- Where: `anthropic_claude_agent_sdk_comprehensive_guide.md:858-876`.
- Problem: The SDK yields the error `ResultMessage` first and raises `ResultError` only afterwards. Because `return message.result` runs on the first `ResultMessage`, the generator is closed before `ResultError` can be raised. Tested: one `query()` call, and the function returned `'API Error: 529 overloaded'` as if it were the answer.
- Fix: Branch on `message.is_error`, `message.terminal_reason` and `message.api_error_status` inside the loop. Alternatively, store the result and let the loop finish so `ResultError` is raised.

**B2. Python production guide: `run_resilient` has the same bug.**
- Where: `anthropic_claude_agent_sdk_production_guide.md:49-68`.
- Problem: It returns the first `ResultMessage`, so the `except ResultError` branches (limit vs retryable) are dead. Tested: one call, error result returned, no retry.
- Fix: Same as B1. Keep the `RETRYABLE_STATUS` logic but apply it to the `ResultMessage` fields, or iterate to completion.

**B3. TS production guide: `runResilient` retries the limit errors it claims to leave alone.**
- Where: `claude_agent_sdk_typescript_production_guide.md:28-58`.
- Problem: The TS iterator throws (`Claude Code returned an error result: …`) after yielding an error result. The `LIMIT_SUBTYPES` early return is therefore never reached, and the thrown error is caught and retried. Tested with `error_max_turns`: 4 CLI processes spawned, then it threw.
- Fix: Inside the loop, return when `message.type === 'result' && (success || LIMIT_SUBTYPES.has(subtype))`. In `catch`, only retry when the last result was transient (`terminal_reason === 'api_error'` or `api_error_status` is in a retryable set).

**B4. TS: wrong claim that "failures inside the agent loop arrive as a result message" and only start-up failures and aborts throw.**
- Where: `claude_agent_sdk_typescript_comprehensive_guide.md:704`, `claude_agent_sdk_typescript_production_guide.md:17-21`, `claude_agent_sdk_typescript_migration_guide.md:89`.
- Problem: The iterator also throws after every error result. As a result, the shared `run()` helper (`comprehensive_guide.md:134-153`, `recipes.md:15-34`) never returns a non-success result, and these branches are unreachable:
  - `text()`'s error branch;
  - `comprehensive_guide.md:572`;
  - `recipes.md:135`;
  - the router fallback to `'general'` (`advanced_multi_agent.md:344`), which throws instead of falling back;
  - `advanced_multi_agent.md:347` `Sorry, that failed`.
- Fix: Document the throw-after-error-result behaviour. Have `run()` catch the error and return the captured result when one was seen, or wrap the routing call in try/catch.

**B5. Python migration guide: false claim about the current `setting_sources` default.**
- Where: `anthropic_claude_agent_sdk_migration_guide.md:28-29, 92-100, 180`.
- Problem: The guide implies settings and `CLAUDE.md` are not loaded by default. In 0.2.170, `setting_sources=None` loads all sources (`types.py` docstring: "When None, all sources are loaded (matches CLI defaults)"). The TS guide states this correctly (`migration_guide.md:85`, `comprehensive_guide.md:766`), and so does the Python comprehensive guide (`:156`).
- Fix: Say that 0.1.0 stopped loading settings by default, but current releases load all sources when `setting_sources` is omitted. Pass `[]` for isolation. Reword the troubleshooting entry at L180.

**B6. Python production guide: "Hooks or `can_use_tool` never fire" when a tool is in `allowed_tools` is wrong for hooks.**
- Where: `anthropic_claude_agent_sdk_production_guide.md:352`.
- Problem: Hooks run first, before allow rules. The guide's own order at `comprehensive_guide.md:404` says so, and the SDK's warning text says "To gate every tool call, use a PreToolUse hook". Getting this wrong misleads readers about a security control.
- Fix: Change the heading to "`can_use_tool` never fires". Add that `PreToolUse` hooks still run and are the way to gate pre-approved tools.

### Major

**M1. Python comprehensive guide: wrong description of `load_timeout_ms`.**
- Where: `anthropic_claude_agent_sdk_comprehensive_guide.md:948`.
- Problem: The guide calls it "CLI start-up timeout". It is actually the timeout for each `session_store.load()`/`list_subkeys()` call during resume.
- Fix: Move it to the session-store row and describe it correctly.

**M2. Python pages: `TodoWrite` listed as a current built-in tool.**
- Where: `anthropic_claude_agent_sdk_comprehensive_guide.md:320`, `anthropic_claude_agent_sdk_diagrams.md:82`.
- Problem: The bundled CLI (2.1.306) uses Task tools (`TaskCreate`/`TaskUpdate`/`TaskGet`/`TaskList`) in SDK sessions, as the TS guide notes (`comprehensive_guide.md:283`, `migration_guide.md:149`).
- Fix: List the Task tools, matching the TS guide.

**M3. Both comprehensive guides: `usage` presented as run totals.**
- Where: `anthropic_claude_agent_sdk_comprehensive_guide.md:288-303` ("authoritative totals") and `claude_agent_sdk_typescript_comprehensive_guide.md:214-228`.
- Problem: Upstream (TS CHANGELOG 0.3.223) documents `usage` as main-loop-only and per turn, with `modelUsage` as the cumulative field for cost accounting.
- Fix: Say so and print totals from `model_usage`/`modelUsage`, as the production pages already do.

**M4. Python diagrams: invented compaction mechanics.**
- Where: `anthropic_claude_agent_sdk_diagrams.md:733-776`.
- Problem: "SUMMARISATION / PRUNING / COMPRESSION" strategies, "Keep recent 5 messages, summarise older 95" and "50% of original size" are not SDK or CLI behaviour.
- Fix: Replace with the real flow: `autoCompactThreshold`, then the `PreCompact` hook, then summarisation, then a `compact_boundary` system message (as in the TS diagrams at L333-374).

**M5. Python diagrams: generic deployment filler.**
- Where: `anthropic_claude_agent_sdk_diagrams.md:807-919`.
- Problem: HPA "Target CPU 70%", "Database password", "Redis cache layer" and "Metrics data" on a PVC have nothing to do with the SDK and contradict the production guide's advice to size by concurrent processes.
- Fix: Reduce to one diagram: worker, CLI subprocess, `$HOME/.claude` volume or `SessionStore`, queue.

**M6. TS diagrams: invented error model and caching advice.**
- Where: `claude_agent_sdk_typescript_diagrams.md:376-433` and `:496-551`.
- Problem: The error "severity" table and the classify/stop/alert flow are invented and not tied to `result` subtypes. The "L1/L2 cache, query Claude, cache result" advice caches agent runs by prompt, which is unsafe for tool-using agents.
- Fix: Map the real failure surfaces (`result` subtypes and `terminal_reason`, the throw after an error result, `AbortError`). Drop the response cache, or replace it with prompt caching notes.

**M7. Frontmatter inconsistent.**
- Problem: `framework` takes three values: `anthropic-claude-sdk` (both `index.mdx`), `anthropic-claude-agent-sdk` (5 Python pages) and `anthropic-claude-agent-sdk-typescript` (8 TS pages). `language` is missing on all 13 non-index pages. Other guides (for example langgraph) use one framework value plus `language` on every page.
- Fix: Use one `framework` value (for example `anthropic-claude-agent-sdk`) on all 15 pages, and add `language: python` or `language: typescript`.

**M8. Sidebar shows only the overview for each language.**
- Where: `astro.config.mjs:258-259`.
- Problem: All 13 subpages are reachable only through index cards; there are no sidebar entries for the comprehensive guide, recipes, production and so on. Other multi-page guides use a collapsed group.
- Fix: Turn each entry into a collapsed group with Overview, Comprehensive guide, Recipes, Production, Migration and Diagrams (plus Middleware, Advanced multi-agent and Streaming server for TS).

**M9. TS middleware: wrong claim that `PostToolUse` cannot replace tool output.**
- Where: `claude_agent_sdk_typescript_middleware.md:56-77`.
- Problem: The section says `PostToolUse` "can add context" and redaction needs a `PreToolUse` deny. But `updatedToolOutput` exists (`sdk.d.ts:2727`, and the Python guide documents it at `comprehensive_guide.md:518`). The page's own table (L15) says "replace tool output". The section titled "Redacting" doesn't redact anything.
- Fix: Show real redaction with `hookSpecificOutput.updatedToolOutput`.

**M10. Coverage gaps (vs. official SDK docs and the CHANGELOGs).**
- Missing in both: handling user questions (`AskUserQuestion` / permission prompts with `suggestions` and `updated_permissions`), image/multimodal input via streaming prompts (mentioned, never shown), a `SessionStore` implementation example (only `InMemorySessionStore` appears), skills and plugins beyond one table row, and hosting patterns (ephemeral container per session, session-store resume across hosts).
- Python has no streaming web server (FastAPI SSE), middleware or advanced multi-agent page, although TS has all three.
- TS has no testing guidance, while Python has Recipe 14.
- Fix: Add a "User input and approvals" section to both comprehensive guides, a Redis/S3 `SessionStore` example, and a Python SSE recipe. Add a TS testing recipe that uses `pathToClaudeCodeExecutable` with a stub, or a mocked `run()`.

### Minor

**m1. Relative internal links in both index pages.**
- Where: Python `index.mdx:30, 36-59, 74, 117`; TS `index.mdx:30, 36-59, 96`.
- Problem: They use relative links (`./anthropic_…/`), against the AGENTS.md rule.
- Fix: Use `/anthropic-claude-agent-sdk-guide/…`.

**m2. Python recipe 4 can crash when cost is missing.**
- Where: `anthropic_claude_agent_sdk_recipes.md:147`.
- Problem: `f"{result.total_cost_usd:.3f}"` raises `TypeError` when the cost is `None` (the field is optional).
- Fix: Use `(result.total_cost_usd or 0)`, as the comprehensive guide does at L189.

**m3. Python production guide: budget enforcement attributed to the SDK.**
- Where: `anthropic_claude_agent_sdk_production_guide.md:106`.
- Problem: It says "`max_budget_usd` is enforced by the SDK"; the CLI enforces it.
- Fix: "enforced by Claude Code".

**m4. Thinking notes name only Opus 5.5 as unable to disable thinking.**
- Where: Python `comprehensive_guide.md:769`; TS `comprehensive_guide.md:634`.
- Problem: Sonnet 5.5 and Fable 5.1 also reject `{type: "disabled"}`.
- Fix: "and none of them accepts disabled thinking except Haiku 5.5 at effort high or below", or simply "do not disable thinking; lower effort".

**m5. TS advanced multi-agent: prompt refers to tools that don't exist.**
- Where: `claude_agent_sdk_typescript_advanced_multi_agent.md:33-36`.
- Problem: The billing specialist is told to use "only the billing tools", but it has `tools: []`.
- Fix: Add an SDK MCP billing tool, or change the prompt.

**m6. TS Dockerfiles have no build step.**
- Where: `claude_streaming_server_express.md:88-98`, `claude_agent_sdk_typescript_production_guide.md:215-231`.
- Problem: They `COPY dist` without building it, and `npm ci --omit=dev` drops `typescript`.
- Fix: Use a multi-stage build that runs `npm ci && npx tsc` in a builder stage.

**m7. Inconsistent titles.**
- `"Claude Agent SDK - Architecture & Flow Diagrams"` (Python, no "(Python)", hyphen), `"Claude Agent SDK Advanced Multi-Agent (TypeScript)"`, `"Claude Agent SDK Middleware (TypeScript)"` and `"Claude Streaming Server (Express)"` sit alongside the "— Comprehensive Guide" pattern.
- Fix: Use `Claude Agent SDK (Python|TypeScript) — <Topic>` throughout.

**m8. Inconsistent file name.**
- `claude_streaming_server_express.md` breaks the `claude_agent_sdk_typescript_*` prefix.
- Fix: Rename it (with a redirect) when convenient.

**m9. Marketing-style closing lines on the diagrams pages.**
- Where: `anthropic_claude_agent_sdk_diagrams.md:923` and `claude_agent_sdk_typescript_diagrams.md:553` ("illustrate the complex interactions…").
- Fix: Delete them.

**m10. Python migration fragments lack imports.**
- Where: `anthropic_claude_agent_sdk_migration_guide.md:77-86, 94-98`.
- Problem: These blocks use `ClaudeAgentOptions` without importing it.
- Fix: Add `from claude_agent_sdk import ClaudeAgentOptions`.

**m11. Python recipe 14: unnecessary asyncio marker.**
- Where: `anthropic_claude_agent_sdk_recipes.md:494-497`.
- Problem: `test_tool_handler_is_callable_directly` is marked `asyncio` but has no awaits. It is harmless.
- Fix: Await `lookup_order.handler` with a mocked `httpx` transport, which would actually test something.

**m12. Python recipe 8 warning is not explained.**
- Where: `anthropic_claude_agent_sdk_recipes.md:275`.
- Problem: Recipe 8 triggers `CanUseToolShadowedWarning` for Read and Grep. This is intended, but the reader isn't told.
- Fix: Add a one-line note that the warning is expected.

#### Top 5 improvements

1. **Fix failure semantics everywhere (B1–B4).** State once per language that the SDK yields the error `result` and then raises or throws. Rewrite `run_with_retry`, `run_resilient`, `runResilient` and the shared `run()` helpers to match, and add an error-path test (the stub-CLI approach in the scratch dir works for both languages).
2. **Correct the false claims (B5, B6, M1, M9, M3)** about the `setting_sources` default, hooks vs `allowed_tools`, `load_timeout_ms`, `updatedToolOutput`, and `usage` vs `modelUsage`.
3. **Normalise frontmatter and navigation (M7, M8, m1, m7).** Use one `framework` value, add `language` to every page, add collapsed sidebar groups, use root-absolute links, and give titles a consistent pattern.
4. **Rewrite both diagrams pages around real mechanics (M4–M6, m9).** Remove the invented compaction strategies, severity tables and response caching, and the K8s/HA filler; show the subprocess, control protocol, permission order and session store.
5. **Close the coverage gaps (M10, M2).** Add user-input/approval handling and a real `SessionStore` example to both languages, a Python SSE server and TS testing recipe for parity, and the Task tools in place of `TodoWrite` in Python.

---

### Assessment: smolagents guide (`src/content/docs/smolagents-guide/`)

Assessed 2026-10-21 against `smolagents` 1.26.0, which is both the version in `src/data/versions.json` and the latest release on PyPI (there has been no release since 2026-05-29; `main` is at 1.27.0.dev0 with CI and docs changes only). Pages read in this order: `index.mdx` → `smolagents_comprehensive_guide.md` → `smolagents_recipes.md` → `smolagents_production_guide.md` → `smolagents_diagrams.md`.

Method: venv at `scratchpad/smolagents/venv` with smolagents 1.26.0 and the extras installed (mcp 1.30, litellm 1.106.2, openinference-instrumentation-smolagents 0.1.43, e2b, docker, ddgs, pandas, fastapi). Every Python block went through three checks: a syntax check, an import check, and an exec harness that swaps every model class for a stub that calls `final_answer`. The behaviour claims (interrupt, max steps, streaming, `final_answer_checks`, remote executors with managed agents, tool validation, `LogLevel`, telemetry spans, MCP over stdio, the production tests) were checked by separate scripts using real classes (`scratchpad/assess-smolagents/scripts/`). Release history was checked against `git log` between the version tags of a clone of huggingface/smolagents. Model IDs were checked against the litellm 1.106.2 model map and against the Hugging Face Hub's inference-provider listing.

#### Scorecard

| # | Criterion | Score | Justification |
|---|-----------|-------|---------------|
| 1 | Correctness of code | 4/5 | All 74 Python blocks parse and every `smolagents` import resolves. Every runnable block passes with a stub model or with real classes. The code failures are in the diagrams (`output_type = "str"` and an `@tool` with no `Args:`, both of which raise) and one stale `load_in_4bit` kwarg. |
| 2 | Currency | 3/5 | Versions use placeholders and `check-versions` is clean. However: `claude-sonnet-4-5`, `gemini-2.5-flash` and `gpt-5-mini` are stale. The library's default model has no working inference provider. `provider="together"` is no longer listed for either Qwen model shown with it. |
| 3 | Accuracy of prose | 3/5 | The comprehensive, recipes and production pages are largely accurate, and every release-history claim checks out. The diagrams page has invented figures ("25% CAGR", "~40% fewer tokens", "3.2 s vs 32 s") and false security claims about the Docker executor. There is also a wrong statement about where tools run under remote executors, and a wrong `LogLevel` mapping. |
| 4 | Coverage | 3/5 | Core API coverage is good. Missing: the `smolagent`/`webagent` CLI, the vision web-browser agent, agentic RAG, async applications, human-in-the-loop plan editing, `ToolCollection.from_mcp`, `agent.visualize()`, and `AgentImage`/`AgentAudio`. The diagrams page is about 900 lines of ASCII and mostly padding. |
| 5 | Structure & navigation | 4/5 | The index has install, a working minimal example and a page map. No links are broken and there are no deep-dive or "Vol." pages. The index uses relative `./` links rather than root-absolute ones. The revision-history rows are tied to placeholders, and the page TOCs duplicate the TOC Starlight already provides. |
| 6 | Consistency | 3/5 | The API style is consistent and British English is used throughout. Four of the five pages have no `language:` in their frontmatter. The production page's HPA scales on CPU, which contradicts its own advice two screens later. The "verified with a stub model" banners claim more than was verified. |
| 7 | Production readiness | 4/5 | Real, current guidance: executor choice, `RunResult` checks, token budgets via `interrupt()`, OpenInference/OTLP, Weave, a scripted-model test pattern, FastAPI plus K8s. Gaps: no streaming endpoint, no egress control on the Docker executor, no async guidance. |

**Overall verdict.** The three prose pages (comprehensive, recipes, production) are good: accurate against 1.26.0, honest about sandboxing, and with code that works. The guide is held back by the diagrams page, which still has invented benchmark and market numbers and false Docker isolation claims despite its "illustrative" banner. The second problem is model choices: the out-of-the-box `InferenceClientModel()` default no longer has a live provider, and several model IDs are a generation or more old.

**Blockers: 4** (diagrams page has invented numbers and wrong behaviour; diagrams page makes false Docker isolation claims; invalid tool code in the diagrams; comprehensive guide misstates where tools run in remote executors). One related blocker sits outside the folder (`quick-start.md`).

#### Code verification results

Blocks counted are fenced `python` blocks. Exec status comes from the stub-model harness; "env" means the block failed only for lack of a network service, credential or file, and the API surface was then checked separately.

| Page | Python blocks | Parse | Imports | Exec pass | Env-only fail | Illustrative (not runnable by design) | Real failures |
|------|---------------|-------|---------|-----------|---------------|----------------------------------------|---------------|
| index.mdx | 1 | 1 | ok | 1 | 0 | 0 | 0 |
| smolagents_comprehensive_guide.md | 36 | 36 | ok | 27 | 7 | 2 (blocks 0, 22: model-written snippets referencing undefined `get_hotel_price` / `get_temperature`) | 0 |
| smolagents_recipes.md | 22 | 22 | ok | 22 | 0 | 0 | 0 |
| smolagents_production_guide.md | 15 | 15 | ok | 13 | 2 | 0 | 0 |
| smolagents_diagrams.md | 0 (code in ASCII boxes) | n/a | n/a | n/a | n/a | n/a | 2 snippets that would raise (see B3) |
| **Total** | **74** | **74** | **all resolve** | **63** | **9** | **2** | **0 in fenced blocks; 2 in diagrams; 1 deprecated kwarg (M7)** |

Env-only failures, each then checked another way:
- Comprehensive block 15 (streaming) failed only because the stub had no `generate_stream`. A stub that streams yields `ChatMessageStreamDelta`, `ToolCall`, `ActionOutput`, `ActionStep` and `FinalAnswerStep`, so the code is correct.
- Block 19 (`load_tool`, `Tool.from_space`, `ToolCollection.from_hub`, `push_to_hub`) failed on a 401. All signatures match.
- Block 20 (MCP with `uvx mcp-server-fetch`) could not reach the server. The same pattern against a local FastMCP stdio server lists the tools and runs to completion.
- Block 30 failed because the stub returns `'42'` rather than a dict. The validation logic is correct.
- Block 32 (Docker daemon), block 33 (Hub authentication), block 34 (`chart.png`) and block 35 (gradio not installed) need the missing service, credential, file or package.
- Production block 7 (OTel) failed in the harness only because the harness patches the model classes. With real classes the instrumentor emits `CodeAgent.run`, `Step 1` and `FinalAnswerTool` spans.
- Production block 8 needs `E2B_API_KEY`.
- Production block 14 needs `weave`. The weave 0.53.12 wheel has `weave/integrations/smolagents`, so the auto-patch claim holds.
- The production `fx_rate` and `ScriptedModel` tests pass with real classes.

Behaviour claims verified true:
- `HfApiModel` is present in 1.18.0 and gone in 1.19.0.
- The default model changed in 1.23.0, in #1813.
- 1.24.0 brought the robust timeout, `FinalAnswerStep` in callbacks, and the `HfApiModel` load compatibility.
- 1.25.0 set `allow_pickle=False` and added the "not a security tool" wording and hub 1.x support.
- 1.26.0 added Exa and removed `WasmExecutor`.
- A remote executor combined with `managed_agents` raises "Managed agents are not yet supported with remote code execution".
- `interrupt()` raises `AgentError("Agent interrupted.")`.
- Running out of steps gives `state == "max_steps_error"`.
- Final-answer checks receive `agent=`.
- `code_block_tags="markdown"` gives triple-backtick fences.
- Names that are not identifiers are rejected.
- A missing `Args:` section raises `DocstringParsingException`.
- An input with a default but no `nullable` fails validation.
- The MCP compatibility note is accurate: a fresh `pip install "smolagents[mcp]==1.26.0"` pulls mcp 2.3.0, and `MCPClient` then fails with `ImportError: cannot import name 'streamablehttp_client'`.

`node scripts/check-versions.mjs | grep smolagents` gives no output (clean). `node scripts/check-links.mjs` reports 1,838 links with 0 broken.

#### Issues

### Blocker

**B1. Invented figures and wrong agent behaviour in the diagrams.**
`smolagents_diagrams.md`, lines 265–266, 431, 447–448, 502, 717, 778–819:
- Lines 265–266 say "CodeAgent completes in 1 LLM call", but the shown code (lines 182–188) never calls `final_answer`, so the run would not end.
- Lines 238 and 258 invent `calculator` and `report` tools. The real terminal tool is `final_answer`.
- "25% CAGR … $2.5T by 2030" (lines 431, 447–448), "3.2 seconds (vs 32 seconds sequentially)" (line 502), "Token efficiency: ~40% fewer tokens" (line 819) and per-call figures like "200 in, 150 out" are invented. The CodeAgent system prompt alone is several thousand tokens.

The banner at line 7 does not excuse a stated "~40%".
*Fix:* remove every number from sections 4, 6 and 8. Rewrite §2 to show `final_answer(...)` in the code and the real `final_answer` tool on the JSON side. Replace §8 with a qualitative comparison that points to the comprehensive guide's "measure with `result.token_usage`" advice.

**B2. False Docker executor isolation claims.**
`smolagents_diagrams.md`, lines 585–589, 539–542:
- The Docker box claims "No access to host files / Network isolated / Resource limits". `DockerExecutor` starts the container with only a port mapping: no network restriction, no memory or CPU limits (it needs network to `pip install` the authorised imports). Limits exist only if you pass them yourself in `container_run_kwargs`.
- Line 125 says "Error occurred? → Stop or retry". In fact errors are recorded and the loop continues.

*Fix:* change the box to "Separate container; network and resources unrestricted unless set via `executor_kwargs={"container_run_kwargs": {...}}`". Change line 125 to "Error → recorded as observation, loop continues".

**B3. Invalid tool code in the diagrams.**
`smolagents_diagrams.md`, lines 284–290:
- `output_type = "str"` fails `Tool` validation with an `AssertionError` (verified); the allowed value is `"string"`.
- `@tool def my_function():` with only `"""Docstring"""` (line 285) gives a tool with no typed inputs and no return annotation, which the decorator does not accept.
- Line 386 ("Capture: analysis variable") and lines 618–621 ("Variables captured") misdescribe the loop. The observation is the printed output plus the `final_answer` value, not the variables.

*Fix:* use `output_type = "string"` and a typed function with an `Args:` section, and describe observations as "print() output / final_answer value".

**B4. Wrong statement about where tools run under remote executors.**
`smolagents_comprehensive_guide.md`, line 889: "The remote executors run the generated code remotely while tools execute where they are defined; tool objects are sent to the sandbox". The sentence contradicts itself, and the first half is false. `RemotePythonExecutor.send_tools` installs each tool's requirements and re-creates the tool from its source code inside the sandbox, so tools execute in the sandbox.
*Fix:* "Tools are rebuilt from their source inside the sandbox and run there, so they must be self-contained (imports inside `forward`, credentials available in the sandbox)."

**Related, outside the folder.** `src/content/docs/quick-start.md`, lines 21–29: the "fastest start" smolagents snippet uses `@tool` with no `Args:` section, which raises `DocstringParsingException`, and calls `CodeAgent(tools=[add_numbers])` with no `model`, which raises a `TypeError`. *Fix:* copy the index minimal example. Line 36 also says "Deep dive:", which should be "Reference:".

### Major

**M1. The library's default model has no live inference provider.**
`index.mdx:96`, `smolagents_comprehensive_guide.md:192, 300, 317`, and most recipes (`model = InferenceClientModel()` at `smolagents_recipes.md:14`). The Hub lists one provider for `Qwen/Qwen3-Next-80B-A3B-Thinking`, novita, and its status is "error". A reader's first `InferenceClientModel()` call is therefore likely to fail.
*Fix:* pass an explicit, currently served `model_id` in the index and the recipes preamble, and add a one-line note that the library default can lose its providers. Point readers to the Hub's "Inference Providers" panel.

**M2. Stale model IDs.**
- `smolagents_comprehensive_guide.md:215`: `anthropic/claude-sonnet-4-5` → `anthropic/claude-sonnet-5-5`.
- `smolagents_comprehensive_guide.md:216`: `gemini/gemini-2.5-flash` → `gemini/gemini-3.8-flash`, which is current in litellm 1.106.2.
- `smolagents_comprehensive_guide.md:214, 235, 726, 911` and `smolagents_recipes.md:14`: `gpt-5-mini` → `gpt-5.4-mini`. litellm lists up to `gpt-5.6` and the `gpt-5.4-mini` family.
- `smolagents_comprehensive_guide.md:195, 236, 247`, `smolagents_production_guide.md:66`: the Qwen2.5-Coder models date from 2024. Consider a current Qwen3 coder model that is actually served.

**M3. `provider="together"` is not a live provider for the models it is paired with.**
`smolagents_comprehensive_guide.md:196` (Qwen2.5-Coder-32B; live providers are nscale and featherless-ai) and `:735` (Qwen2.5-72B; live providers are featherless-ai and deepinfra).
*Fix:* use a provider the Hub lists as live, or show `provider="auto"`.

**M4. Revision-history rows are tied to placeholders.**
`index.mdx:108` and `smolagents_comprehensive_guide.md:946` use `%%versions-checked%% | %%version=smolagents%%` as the date and version of a past revision. On the next bump these rows will silently claim the re-verification happened against the new version. AGENTS.md says revision-history rows use literal versions.
*Fix:* write `2026-10-21 | 1.26.0`. Remove the duplicate "2026-05-29 | 1.26.0 | Version bump" row, or merge the two.

**M5. The production page contradicts itself on autoscaling.**
`smolagents_production_guide.md`, lines 480–483 vs 494. The HPA scales on CPU at 70%, but line 494 says "Scale pods on concurrent requests or latency rather than CPU, since runs mostly wait on the model".
*Fix:* switch the HPA to a custom or external metric (in-flight requests, via KEDA or Prometheus Adapter), or note that the CPU metric is a placeholder.

**M6. Wrong `LogLevel` mapping.**
`smolagents_comprehensive_guide.md:352` says "LogLevel: 0 off, 1 info, 2 debug". The real values are `OFF=-1, ERROR=0, INFO=1, DEBUG=2`. `smolagents_production_guide.md:251` ("`verbosity_level=0` to silence the Rich console output") is consequently half-true: errors still print.
*Fix:* "-1 off (`LogLevel.OFF`), 0 errors only, 1 info, 2 debug", and use `LogLevel.OFF` in production.

**M7. Stale quantisation kwarg.**
`smolagents_comprehensive_guide.md:251`: `model_kwargs={"load_in_4bit": True}` relies on the deprecated `load_in_4bit` argument to `from_pretrained`. Recent Transformers requires `quantization_config=BitsAndBytesConfig(load_in_4bit=True)`.
*Fix:* `model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_4bit=True)}`.

**M8. Missing core features from the official docs** (installed 1.26.0 package, `docs/source/en`):
- The `smolagent` and `webagent` CLIs (`pyproject [project.scripts]`) are not mentioned. *Fix:* add an "Installation & Setup → CLI" subsection.
- The vision web-browser agent (`vision_web_browser.py`, helium) is absent. *Fix:* add a recipe under Web Research.
- Agentic RAG (official example) is absent. *Fix:* add a retriever-tool recipe.
- Async applications (the official `async_agent` example) are absent; production only says "agents are synchronous". *Fix:* add a short section with `anyio.to_thread` / Starlette.
- Human-in-the-loop plan editing (official `plan_customization` example: a `PlanningStep` callback plus `interrupt()` and `run(reset=False)`). *Fix:* a recipe.
- `ToolCollection.from_mcp`, `agent.visualize()`, `AgentImage`/`AgentAudio`, and `PromptTemplates` are not covered. *Fix:* one line each in the relevant sections.

**M9. The diagrams page is mostly padding and has broken box art.**
`smolagents_diagrams.md`, about 890 lines:
- Many boxes have misaligned or missing right borders (e.g. lines 29–31, 405–460, 727–770); the §7 decision tree is unreadable.
- Marketing wording: "Perfect for…" at 529/535/542/549, "Auto-scaling" at 527/534, "100+ providers" at 45/861.
- Closing fluff at line 890, and "comprehensive" at line 11.

*Fix:* cut to 4–5 diagrams (agent loop, CodeAgent vs ToolCallingAgent step, managed-agent call, executor choice), or redraw them with Mermaid/SVG.

### Minor

- **m1.** Frontmatter is missing `language: python` in `smolagents_comprehensive_guide.md:1–5`, `smolagents_recipes.md:1–5`, `smolagents_production_guide.md:1–5` and `smolagents_diagrams.md:1–5`. *Fix:* add it.
- **m2.** `index.mdx:38–70`: the LinkCards use relative `./…` hrefs, but AGENTS.md asks for root-absolute links. *Fix:* use `/smolagents-guide/smolagents_comprehensive_guide/#…`.
- **m3.** `index.mdx:15` and `:23–25`: the current version appears twice (banner and "Version" card). *Fix:* drop the card, or swap it for "Licence/Repo".
- **m4.** `smolagents_comprehensive_guide.md:7` ("Every example … was run … with a stub model") and `smolagents_recipes.md:7` ("network-dependent tools are stubbed where noted") overclaim. Hub, Docker, Gradio and image blocks cannot run with a stub, and no recipe notes any stubbing. *Fix:* "Examples were executed with a stub model where they need no external service".
- **m5.** `smolagents_comprehensive_guide.md:435`: the error hierarchy is shown flat. `AgentToolCallError` and `AgentToolExecutionError` subclass `AgentExecutionError`. *Fix:* show the nesting.
- **m6.** `smolagents_comprehensive_guide.md:682`: "`agent.interrupt()` … stops the run after the current step" should add "by raising `AgentError`" (verified). Do the same at `smolagents_production_guide.md:217`.
- **m7.** `smolagents_comprehensive_guide.md:412`: the stream also yields `ToolCall` and `ActionOutput` events. *Fix:* mention them, or filter them in the loop.
- **m8.** `smolagents_comprehensive_guide.md:594`: `MCPClient` emits a FutureWarning when `structured_output` is unset, and the warning still names 1.25 as the switch release. *Fix:* recommend always passing `structured_output=` explicitly, and use it in the line 589 example.
- **m9.** `smolagents_comprehensive_guide.md:548`: `ApiWebSearchTool` defaults to the Brave Search API endpoint. *Fix:* say "Brave by default; any JSON search API via `endpoint`".
- **m10.** `smolagents_comprehensive_guide.md:85`: the `torch` extra is missing from the "Other extras" list.
- **m11.** `smolagents_comprehensive_guide.md:636`: `"requests"` appears in `additional_authorized_imports` next to a paragraph saying the local executor is not a sandbox. *Fix:* drop `requests`, or add "only with a remote executor".
- **m12.** `smolagents_comprehensive_guide.md:11–27`, `smolagents_recipes.md:21–32` and `smolagents_production_guide.md:11–22` have hand-written "Table of Contents" sections that duplicate Starlight's on-page TOC. *Fix:* remove them; the index already maps the pages.
- **m13.** `src/data/frameworks.ts:31`: the tagline "~1,000 lines" is inaccurate (`agents.py` alone is 1,813 lines; the package is about 15k). *Fix:* "Minimalist, code-first agents".
- **m14.** `index.mdx:35`, "Zero → Hero": mildly hype-flavoured for the house style. *Fix:* "Learning path".
- **m15.** `smolagents_comprehensive_guide.md:222`: `api_version="2024-10-21"` is an old Azure API version. *Fix:* use the current GA version, or a placeholder comment.

#### Top 5 improvements

1. **Rewrite or cut `smolagents_diagrams.md`** (B1–B3, M9). Remove every invented number, fix the Docker isolation claims and the invalid tool snippet, and reduce it to a handful of accurate diagrams. This is the single biggest quality drag.
2. **Fix model selection everywhere** (M1–M3). Use an explicit, currently served `model_id` and a live provider in the index, the recipes preamble and the comprehensive examples. Update to `claude-sonnet-5-5`, `gemini-3.8-flash` and `gpt-5.4-mini`, and say how to check a model's live providers on the Hub.
3. **Correct the remote-executor and logging facts** (B4, M6, m6). Say tools are rebuilt and run inside the sandbox, give the right `LogLevel` values, and note that `interrupt()` raises `AgentError`.
4. **Fill the coverage gaps with recipes** (M8): CLI, vision web-browser agent, agentic RAG, async serving, and human-in-the-loop plan editing. These are the official docs' own examples.
5. **Tidy structure and consistency** (M4, M5, m1–m4, m12). Use literal revision-history rows, a consistent HPA metric, `language:` frontmatter, root-absolute index links, accurate verification banners, and no hand-written TOCs. Also fix the broken smolagents snippet in `quick-start.md`.

---

### Assessment: Mistral Agents API guide

Scope: `src/content/docs/mistral-agents-api-guide/` (10 Python pages + index, 4 TypeScript pages). Assessed 2026-10-21 against `mistralai` 3.3.0 (PyPI latest, released 2026-10-20) and `@mistralai/mistralai` 3.1.0 (npm latest, released 2026-10-20), both matching `src/data/versions.json`. Paths below are relative to the guide folder unless they start with `src/` or the repo root.

Scratch work: `/tmp/claude-0/-home-user-AgentGuides/55b0d51f-358a-5617-8094-250bd4df3d90/scratchpad/assess-mistral/` (Python mock transport `mock.py`, runner `run.py`, TS harness `ts/` with `mockfetch.mjs`).

#### Scorecard

| # | Criterion | Score | Justification |
|---|---|---|---|
| 1 | Correctness of code | 4 | All 75 Python blocks parse, import and match signatures; 72 run against a mocked API. All 37 TS blocks pass `tsc --strict` and 35 run. Two real defects: an MCP server/launch mismatch that hangs, and a NameError when the handbook folder is empty. |
| 2 | Currency | 4 | `check-versions` is clean and every model ID is current (stale IDs appear only in retirement notes). The Python pages still show two deprecated SDK methods (`agents.list`, `connectors.call_tool`) as current; the TS pages flag both correctly. |
| 3 | Accuracy of prose | 3 | Most claims check out against the SDK source and Mistral's docs (v3 migration, credential order, handoff modes, moderation defaults, release dates). But the OpenTelemetry claim is false (checked by running it), the Kubernetes token advice is wrong, beta/preview status is never stated, and the diagrams page invents internals. |
| 4 | Coverage | 3 | Agents, versions and aliases, conversations, streaming, built-in tools, libraries, connectors, RunContext, MCP, approvals and guardrails are all covered well. Gaps: the Python connector-approval flow (`tool_confirmations`), Mistral's own observability and evaluation APIs, testing, cost, and a pointer to Workflows/Skills. The 1,194-line diagrams page is padding. |
| 5 | Structure and navigation | 3 | The index has install, a minimal example and a page map, and links are clean. But there is a per-release history page (AGENTS.md forbids these), deployment/CI content appears three times, the index Reference grid repeats the Zero→Hero grid, and the diagrams page reads as "Comprehensive … Part N". |
| 6 | Consistency | 4 | API style, terminology and British English are steady across Python and TS. Problems: `framework: mistral` on index vs `mistral-agents-api` everywhere else, `language` missing on the Python topic pages, and small table mismatches between the Python and TS pages. |
| 7 | Production readiness | 3 | Real, specific guidance on retries and idempotency, version aliases set in a deploy step, ownership checks, service-account tokens, a circuit breaker and per-user rate limits. Missing: how to test (it says "client mocked" but never shows it), cost control, and correct telemetry set-up. |

**Verdict.** This is one of the more trustworthy guides: nearly every example runs against the current SDK, and the TypeScript half is particularly careful. The main faults are one false observability claim, an MCP example that hangs if you follow the prose, deprecated Python calls the TS pages already flag, and structural leftovers (history page, padded diagrams page, triplicated deployment snippets). Fixing those would take it to 4–5 across the board.

#### Code verification

How: Python blocks were extracted and checked for syntax (`chksyntax.py`), imports (`chkimports.py`), keyword arguments against `inspect.signature` (`kwcheck.py`) and tool-dict shapes against the SDK's pydantic request models (`toolcheck.py`). Each page was then run in order in one namespace against an `httpx2.MockTransport` API mock, with scripted function-call, handoff and deferred-approval turns. TypeScript blocks were extracted into modules, checked with `tsc --strict` (TS 7.0.2, NodeNext), emitted, and run under Node 22 with a global `fetch` mock. Bash, YAML, Dockerfile and SQL blocks were read but not executed.

### Python (`mistralai` 3.3.0)

| Page | Blocks | Static (syntax/imports/kwargs/tool shapes) | Run vs mock | Notes |
|---|---|---|---|---|
| index.mdx | 1 | 1/1 | 1/1 | Minimal example runs (start + append). |
| mistral_agents_api_comprehensive_guide.md | 19 | 19/19 | 19/19 | Manual function loop, RunContext loop, handoff, structured output, guardrails, retry/404 path all ran. |
| mistral_agents_api_connectors_guide.md | 11 | 11/11 | 11/11 | `ToolFileChunk` download wrote `chart.png`. Block at L143 raises NameError if `handbook/` has no PDFs (see Minor). |
| mistral_agents_api_orchestration_guide.md | 13 | 13/13 | 12/13 | L213 fails: `user_is_verified` is undefined (placeholder). Manager-worker loop, server-side handoff, fan-out and queue ran. |
| mistral_agents_api_mcp_guide.md | 7 | 7/7 | 5/5 run, 2 skipped | L92 (npx filesystem server, blocked by proxy TLS) and L150 (remote server) skipped. The RunContext + `MCPClientSTDIO` path was re-run with the page's own `tickets_server.py` and works only after changing `transport` to `"stdio"` (Blocker 2). Deferred approval `to_dict`/`from_dict`/`confirm()` round trip works. |
| mistral_agents_api_advanced_python.md | 6 | 6/6 | 6/6 | Needed `mistralai[telemetry]` for L142. Runs, but does not do what the prose says (Blocker 1). |
| mistral_agents_api_production_guide.md | 7 | 7/7 | 7/7 | `observed_start`, `guarded_start` and the token bucket were also exercised. |
| mistral_agents_api_recipes.md | 10 | 10/10 | 10/10 | Also called `stream_answer`, `branch_from_first_message`, `start_with_retry` (incl. 404), `safe_append` and the CLI `main()` with piped stdin. |
| mistral_streaming_server_fastapi.md | 1 | 1/1 | 1/1 | Served with uvicorn and curled: start/tool/delta/done SSE events, the append path, and the 400 on empty `q`. |
| may_2025_launch_updates.md, mistral_agents_api_diagrams.md | 0 | — | — | No code. |
| **Total** | **75** | **75/75** | **72 pass, 1 fail, 2 skipped** | |

### TypeScript (`@mistralai/mistralai` 3.1.0)

| Page | Blocks | `tsc --strict` | Run vs mock | Notes |
|---|---|---|---|---|
| typescript/index.mdx | 1 | 1/1 | 1/1 | |
| typescript/agents-and-conversations.md | 13 (incl. common.ts) | 13/13 | 12/12 | `listPages`, versions/aliases, restart, streaming, Zod structured output, `chat.parse`, guardrails. |
| typescript/tools-and-orchestration.md | 14 | 14/14 | 12/14 | L533 `connectors.mcpClient()` correctly calls `/v1/connectors-gateway/deepwiki/mcp` (a gateway can't be mocked without a live MCP server). L559 stdio bridge skipped (needs `tickets-server.js`). |
| typescript/production.md | 9 | 9/9 | 9/9 | SSE server ran on a free port and was curled; `MistralCore` standalone function, hooks and limiter all ran. |
| **Total** | **37** | **37/37** | **35 pass, 2 not runnable offline** | |

### Other checks

- `node scripts/check-versions.mjs | grep mistral`: clean.
- `node scripts/check-links.mjs`: 0 broken (287 files, 1,838 links).
- No "deep dive", "Vol. N", TODO or placeholder text. No page body starts with `# `.
- Model IDs used: `mistral-medium-latest`, `mistral-small-latest`, `mistral-medium-3-5`, `mistral-large-2512`, `mistral-large-4`, `mistral-small-2603`, `ministral-8b-2512`, `mistral-moderation-2603`. All are in Mistral's current models list. Retired IDs (`mistral-medium-2505`, `-2508`, `mistral-moderation-2411`) appear only in deprecation or history text. **No stale model IDs.**

#### Issues

### Blocker

1. **mistral_agents_api_advanced_python.md:140**. The page says the SDK "records spans for its HTTP calls when your process has a tracer provider configured". This is false in 3.3.0. Telemetry is opt-in (`TracingHook.before_request` → `configure_telemetry_for_hook` returns False when `MISTRAL_SDK_TELEMETRY` is unset). Running the block with an in-memory exporter produced only the user's own `answer_weather_question` span. After `configure_telemetry(client, provider="global")` the SDK's `chat mistral-small-latest` span appeared. **Fix:** add `from mistralai.extra.observability import configure_telemetry` and `configure_telemetry(client, provider="global")` after `set_tracer_provider` (or say `MISTRAL_SDK_TELEMETRY=global`). Note that redaction applies only in `"dedicated"` mode, so wrap the exporter in `RedactingSpanExporter`. Repeat the correction at **mistral_agents_api_production_guide.md:176**.
2. **mistral_agents_api_mcp_guide.md:276 and :279**. `tickets_server.py` ends with `server.run(transport="streamable-http")`. The next sentence then says to run it locally with `MCPClientSTDIO(... args=["tickets_server.py"])`. Done as written, the server binds HTTP on :8000 and the stdio client hangs forever in `initialize` (reproduced). **Fix:** make the default `server.run()` (stdio), with a comment `# or transport="streamable-http" to deploy as a connector`. Or change L279 to "change the transport to `stdio` first".

### Major

3. **mistral_agents_api_comprehensive_guide.md:127**. `client.beta.agents.list(...)` is `@deprecated` in 3.3.0 ("agent sharing not supported… Use list_pages instead") and emits a DeprecationWarning. The TS page (typescript/agents-and-conversations.md:72-79) already uses `listPages()`. **Fix:** show `client.beta.agents.list_pages(page_size=50, name=...)` with `page_token` pagination and note that `list()` is deprecated.
4. **mistral_agents_api_connectors_guide.md:226-234**. `client.beta.connectors.call_tool(...)` is shown as the way to test, but it is `@deprecated` ("will be removed"). The TS page (tools-and-orchestration.md:531) says so and uses `mcpClient()`. **Fix:** replace it with `async with client.beta.connectors.mcp_client(connector_id_or_name="deepwiki") as session: await session.call_tool(...)` and mention `ConnectorsGatewayError`.
5. **may_2025_launch_updates.md (whole page), index.mdx:61**. This is a per-release history page ("May 2025 Launch and What Changed Since"), which AGENTS.md rules out. Its frontmatter also has no `language`. **Fix:** move the "What changed since" table into comprehensive §14 or the index revision history, keep the launch SimpleQA sentence in connectors §3, delete the page, and remove the index card.
6. **mistral_agents_api_mcp_guide.md:242** (and connectors_guide.md:20, :295). For registered connectors the page says only "put the same key in the connector tool's `tool_configuration`". It never shows how to approve: find the `function.call` entry with `confirmation_status == "pending"`, then `conversations.append(conversation_id=..., tool_confirmations=[{"tool_call_id": ..., "confirmation": "allow"}])`. TS covers this fully at typescript/tools-and-orchestration.md:236-277. **Fix:** add the Python equivalent as §3.1.
7. **Status never stated (index.mdx:13, connectors_guide.md:9)**. The Agents and Conversations APIs are still under `client.beta`, and Mistral's docs describe Connectors as "a Public Preview feature". No current page says either. **Fix:** add one line to the index aside and the connectors intro: "Agents/Conversations: beta; Connectors: public preview; shapes may change."
8. **mistral_agents_api_diagrams.md:1-1194**. The page is padded and partly invented:
   - The title is "Comprehensive Visual Architecture Guide" and the first heading is "Part 1: Comprehensive Diagrams" (L11).
   - L39 shows an `agent_id … status: active` field that does not exist on `Agent`.
   - L66-68 says "Old conversations still accessible and replayable" after an agent is deleted, which is unverified.
   - L928-975 ("LLM Forward Pass … Attention passes"), L979-1035 ("MISTRAL API GATEWAY … Token bucket algorithm") and L1147+ (Redis/PostgreSQL deployment) describe internals. The intro calls them illustrations, but they add nothing for a reader of this API.

   **Fix:** keep Parts 1–3, drop Parts 4–7, remove the `status` field and the deletion claim, and retitle to "Mistral Agents API: Diagrams".
9. **Deployment content triplicated**. A Dockerfile and/or K8s and/or GitHub Actions appear at advanced_python.md:160-227, mistral_streaming_server_fastapi.md:73-145 and production_guide.md:258-306. **Fix:** keep one set in production §8–9 and link to it from the other two pages.
10. **mistral_agents_api_advanced_python.md:203**. "To use a projected service-account token instead of an API key, mount it as a file and set `MISTRAL_SA_TOKEN_PATH`". A Kubernetes projected SA token is not a Mistral credential. `MISTRAL_SA_TOKEN_PATH` expects a Mistral service-account token (MIGRATION.md, "Service-Account Authentication"). **Fix:** "mount the Mistral service-account token from a Secret as a file and point `MISTRAL_SA_TOKEN_PATH` at it".
11. **Testing missing from production pages (production_guide.md:291)**. The CI job runs "unit tests with the client mocked", but no page shows how. **Fix:** add a short "Testing" section that injects `client=httpx2.Client(transport=httpx2.MockTransport(handler))` and scripts a `function.call` then a `message.output` turn (the approach this assessment used). Point TS readers to `new HTTPClient({ fetcher })` (typescript/production.md:189 mentions it but shows no example).
12. **Observability and cost thin (production_guide.md:148-178)**. There is no mention of Mistral's own tracing and evaluation APIs (`client.beta.observability` traces/evaluations/judges/datasets, the `configure_telemetry` dedicated mode with redaction), and no cost guidance beyond "watch usage". Image generation is the only tool given a billing note. **Fix:** add a "Cost" subsection (per-tool billing, `connector_tokens`/`connectors` usage fields, `store=False`, Batch API) and a paragraph on `configure_telemetry`.
13. **Frontmatter inconsistency**. index.mdx:4 has `framework: mistral`, while all other 14 pages have `framework: mistral-agents-api`. The 10 Python topic pages (for example comprehensive_guide.md:1-5) have no `language: python`. **Fix:** use one framework value (`mistral-agents-api`, or `mistral` to match frameworks.ts `id`) and add `language: python` to every Python page.

### Minor

14. **mistral_agents_api_connectors_guide.md:152-161**. `doc` is used after the upload loop: NameError when `handbook/` has no PDFs, and only the last document is checked. **Fix:** collect `docs` and poll each one, as recipes.md:189-195 does.
15. **mistral_agents_api_connectors_guide.md:161**. The comment lists only four `process_status` values. The SDK also has `waiting_for_capacity`, `missing_content`, `self_managed` and `noop` (TS page L164 notes these). **Fix:** "… `done` or `error`, among others".
16. **comprehensive_guide.md:264, connectors_guide.md:17**. The `document_library` "extra fields" omit `tool_configuration`, which the SDK model has and typescript/tools-and-orchestration.md:25 lists. **Fix:** add it.
17. **comprehensive_guide.md:439**. "`HTTPValidationError` … `SDKError` for everything else" omits `ResponseValidationError`, `NoResponseError` and `ObservabilityError`. **Fix:** list them, as typescript/production.md:133-140 does.
18. **orchestration_guide.md:225**. `user_is_verified()` is undefined, so the block fails as pasted. **Fix:** add a stub `def user_is_verified() -> bool: return False  # your own check`, as TS L345 does.
19. **comprehensive_guide.md:95, typescript/agents-and-conversations.md:36**. "Pin the dated API name", but the recommended default `mistral-medium-3-5` is not dated. **Fix:** say "versioned API name (`mistral-medium-3-5`, `mistral-small-2603`)".
20. **index.mdx:49-62**. The "Reference" grid repeats 9 of the 11 Zero→Hero links. **Fix:** drop it, or keep only Recipes and Diagrams.
21. **index.mdx:109**. The revision history says `prompt_cache_key` was documented, but no current page mentions it (it now exists on `agents.complete` requests). **Fix:** document it in comprehensive §10/§12 or reword the history row.
22. **comprehensive_guide.md:378, orchestration_guide.md:195**. Neither says that `handoff_execution="server"` is the default (Mistral's handoffs doc). **Fix:** add "(the default)".
23. **may_2025_launch_updates.md:19 vs connectors_guide.md:135**. The same vendor SimpleQA figures are quoted at different precision (22.08% vs 22%). **Fix:** keep one copy (connectors §3) once item 5 is done.
24. **mistral_agents_api_diagrams.md:554**. "initialized" should be "initialised".
25. **mistral_agents_api_advanced_python.md:2**. The title "Mistral Agents API Advanced (Python)" doesn't match the naming of the other pages, and the page is mostly about `chat.complete`. **Fix:** retitle "Mistral Agents API: Chat Completions and Advanced Python".
26. **src/data/frameworks.ts:223**. The tagline "5 built-in connectors" doesn't match the guide, which lists six tool types and uses "connector" for registered MCP servers. **Fix:** "Hosted agents with built-in tools, MCP connectors and handoffs."
27. **astro.config.mjs:295-301**. The sidebar links only the Python and TS overviews, so the 9 Python topic pages and 3 TS pages are reachable only through index cards. This is the site-wide convention, but the TS pages' `sidebar.label`/`order` frontmatter has no effect. **Fix (optional):** list the topic pages under each language, or drop the unused sidebar frontmatter.
28. **Coverage pointers**. There is no mention of agent deployment to Le Chat (`deployment_chat` on `update`/`list_pages`), the `reasoning_effort` completion argument, Skills/Prompts (`client.beta.skills`, `client.beta.prompts`), or Workflows/Voice agents as neighbouring products. **Fix:** add a short "Related Mistral features" list to the comprehensive guide's overview.

#### Top 5 improvements

1. **Fix the two blockers.** Correct the OpenTelemetry set-up (`configure_telemetry(client, provider="global")` plus `RedactingSpanExporter`) on the advanced and production pages, and make `tickets_server.py` default to stdio so the MCP walkthrough works end to end.
2. **Bring the Python pages up to the TS pages' standard on deprecations and approvals.** Use `list_pages` and `connectors.mcp_client`, and add the Python `tool_confirmations` approval flow for connector tools.
3. **Remove the structural leftovers.** Fold the May 2025 page into comprehensive §14 and delete it, cut the diagrams page to its accurate Parts 1–3, and consolidate the three Dockerfile/K8s/CI copies into the production guide.
4. **Make the production guide complete.** Add a Testing section (an `httpx2.MockTransport` example and the TS `fetcher` equivalent), a Cost section, and Mistral's built-in observability and evaluations.
5. **State status and tidy metadata.** Say plainly that Agents/Conversations are beta and Connectors are public preview, unify `framework:`, add `language: python`, and fix the small table and wording mismatches (items 15–22).

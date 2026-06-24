# PR Conflict Analysis — 2026-06-24

**Repository:** codehalwell/agentguides  
**Analysed:** 2026-06-24  
**Open PRs reviewed:** 4  
**Conflicts found:** 0

---

## Open Pull Requests

| # | Title | Author | Branch | Status | Mergeable |
|---|-------|--------|--------|--------|-----------|
| [#229](https://github.com/CodeHalwell/AgentGuides/pull/229) | Add Microsoft Agent Framework Python Class Deep Dives Vol. 22 (agent-framework 1.9.0) | CodeHalwell | `claude/relaxed-clarke-ueif8r` | Open | ✅ Clean |
| [#228](https://github.com/CodeHalwell/AgentGuides/pull/228) | Add PydanticAI Class Deep Dives Vol. 25 (pydantic-ai v2.0.0) | CodeHalwell | `claude/loving-johnson-qwk345` | Open | ✅ Clean |
| [#227](https://github.com/CodeHalwell/AgentGuides/pull/227) | Add Google ADK Class Deep-dives Vol. 26 (google-adk==2.3.0) | CodeHalwell | `claude/gracious-clarke-a3jwn1` | Open | ✅ Clean |
| [#226](https://github.com/CodeHalwell/AgentGuides/pull/226) | Add LangGraph Class Deep-dives Vol. 24 (langgraph==1.2.6) | CodeHalwell | `claude/gracious-tesla-viso96` | Open | ✅ Clean |

All 4 PRs branch from the same base commit on `main` (`c44e3873`).

---

## File Change Map

Each PR adds exactly 2 files: one new guide document and one update to its framework's `index.mdx`.

| PR | New guide file | Modified index |
|----|---------------|----------------|
| #229 | `microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v22.md` | `microsoft-agent-framework-guide/python/index.mdx` |
| #228 | `pydanticai-guide/pydantic_ai_class_deep_dives_v25.md` | `pydanticai-guide/index.mdx` |
| #227 | `google-adk-guide/python/google_adk_class_deep_dives_v26.md` | `google-adk-guide/python/index.mdx` |
| #226 | `langgraph-guide/python/langgraph_class_deep_dives_v24.md` | `langgraph-guide/python/index.mdx` |

All paths are relative to `src/content/docs/`.

---

## Conflict Analysis

**Result: No conflicts — current or latent.**

- GitHub reports `mergeable_state: "clean"` for all 4 PRs.
- Each PR writes to a **completely separate** framework directory. There is no file overlap between any two PRs.
- The 4 `index.mdx` files modified are in 4 different directories (`microsoft-agent-framework-guide/python/`, `pydanticai-guide/`, `google-adk-guide/python/`, `langgraph-guide/python/`).
- These PRs can be merged in any order without triggering conflicts.

---

## PR Summaries

### PR #229 — Microsoft Agent Framework Vol. 22 (+1504 lines, 7 commits)
Documents 10 previously undocumented class groups from `agent-framework==1.9.0`:
exception hierarchy, `WorkflowInterrupted`/`get_run_context()`, instrumentation helpers, OTel span factories, `WorkflowState` (PowerFx), HTTP request layer, HITL external-input pattern, MCP tool dispatch, security label algebra, and quarantined LLM utilities.

### PR #228 — PydanticAI Vol. 25 (+1607 lines, 5 commits)
Documents 10 `pydantic-ai v2.0.0` class groups including the headline parallel tool execution feature (`ToolManager`/`ParallelExecutionMode`), new `Agent` methods (`to_web()`, `to_cli()`, `run_stream_events()`), Direct API, `format_as_xml`, common tools (`DuckDuckGoSearchTool`, `TavilySearchTool`, `ExaToolset`), capability/registry patterns, `ModelProfile` v2.0.0, `FunctionModel`, and `AgentRun` complete API.

### PR #227 — Google ADK Vol. 26 (+1483 lines, 8 commits)
Documents 10 `google-adk==2.3.0` class groups: `LangGraphAgent` (with checkpointer routing detail), `PubSubToolset`, `SpannerToolset`/`SpannerVectorStoreSettings`, `LongRunningFunctionTool`, `ContextCacheConfig`, `LlmEventSummarizer`, `ToolConfirmation`, `ToolboxToolset`, `DynamicNodeScheduler` (3-case algorithm), and `FileArtifactService`. Also backfills missing Vol. 24 and Vol. 25 entries in the `index.mdx` Jump-to-topic and Reference sections.

### PR #226 — LangGraph Vol. 24 (+1054 lines, 3 commits)
Documents 10 `langgraph==1.2.6` internal class groups: debug payload TypedDicts, `_InjectedArgs`/`_DirectlyInjectedToolArg`, deterministic task ID algorithms (UUID5 vs XXH3-128), deprecation warning hierarchy (including new `LangGraphDeprecatedSinceV11`), `DeprecatedKwargs`, `InvalidModuleError`, `_TasksLifecycleBase`, `_DeltaSnapshot`, `CacheKey`, and `_ToolCallRequestOverrides`.

---

## Recommendation

All PRs are ready to merge with no conflict risk between them. Suggested merge order (oldest first to keep history clean):

1. #226 (LangGraph Vol. 24) — smallest, 3 commits
2. #227 (Google ADK Vol. 26) — 8 commits
3. #228 (PydanticAI Vol. 25) — 5 commits
4. #229 (Microsoft Agent Framework Vol. 22) — 7 commits

Each can be merged independently in any order.

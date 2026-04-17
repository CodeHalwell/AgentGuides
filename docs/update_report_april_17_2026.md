# Agent Guides Update Report — April 17, 2026

**Summary**: Incremental update following the April 16, 2026 major release cycle. Covers version bumps discovered after the April 16 publication, code-level fixes for deprecated patterns, addition of revision history sections to all comprehensive guides, and propagation of version updates across all supplementary files (recipes, production guides, diagrams, GUIDE_INDEX files).

---

## New Versions Discovered (Post-April 16, 2026)

| Framework | Previous (April 16) | New (April 17) | Change |
|-----------|--------------------|--------------------|--------|
| PydanticAI | 1.83.0 | **1.84.0** | Fixes exponential-time regex in Google FileSearchTool |
| Anthropic Claude Agent SDK (Python) | 0.1.59 | **0.1.60** | Extended thinking config, effort field, file checkpointing, session rewind |
| OpenAI Agents SDK (Python) | 0.14.x | **0.14.1** | Patch release stabilising SandboxAgent |

---

## 1. `versions.json`

Updated three framework versions:

- `pydantic-ai`: `1.83.0` → `1.84.0`
- `anthropic-claude-agent-sdk-python`: `0.1.59` → `0.1.60`
- `openai-agents-sdk-python`: `0.14.x` → `0.14.1`

Updated notes for several frameworks to include newly documented features (QuickResearchTool, CVE fixes, native providers, etc.).

---

## 2. PydanticAI — `pydantic_ai_comprehensive_guide.md`

**Version**: `1.14.1` → `1.84.0`

- Updated header and version lines
- Added **Advanced Features (April 2026)** section covering:
  - `EvaluationReport` API for LLM-based evaluation with `pass_rate` and `mean_score`
  - `defer_loading=True` for lazy model initialisation
  - `ThreadExecutor` for sync-in-async tool calls (automatic thread pool wrapping)
  - `CaseLifecycle` hooks for state machine patterns (`on_start`, `on_tool_call`, `on_complete`, `on_error`)
- Added **Revision History** table (v1.84.0, v1.83.0, v1.20.0)

---

## 3. CrewAI — `crewai_comprehensive_guide.md`

**Version**: `1.4.1` → `1.14.0`

- Updated header
- Added **Breaking Change** notice for `CodeInterpreterTool` removal (raises `ImportError`)
- Added **Checkpoint System (v1.14.0+)** section: `CheckpointConfig`, `SqliteProvider`, `crewai checkpoint list/info`
- Added **Native OpenAI-Compatible Providers** section: OpenRouter, DeepSeek, Ollama, vLLM, Cerebras
- Fixed pinned version: `uv pip install crewai==1.4.1` → `crewai==1.14.0`
- Updated `crewai_flows_guide.md`: version references `1.4.1` → `1.14.0`
- Added **Revision History** table (v1.14.0, v1.13.0, v1.5.0)

---

## 4. SmolAgents — `smolagents_comprehensive_guide.md`

**Version**: `1.22.0` → `1.24.0`

- Updated header
- Added Python 3.10+ requirement note in installation section
- Added `InferenceClientModel` vs deprecated `HfApiModel` note in model configuration section
- Added **Revision History** table (v1.24.0, v1.23.0)

---

## 5. Anthropic Claude Agent SDK (Python) — `anthropic_claude_agent_sdk_comprehensive_guide.md`

**Version**: `0.1.6` → `0.1.60`

- Updated header and installation pin
- Added **Extended Thinking Configuration (v0.1.60+)** section:
  - `ThinkingConfigAdaptive` — Claude decides when to use extended thinking
  - `ThinkingConfigEnabled(budget_tokens=N)` — always-on extended thinking
  - `ThinkingConfigDisabled()` — disable thinking entirely
  - `effort` field: `"low"`, `"medium"`, `"high"`, `"max"` with token budget table
- Added **File Checkpointing and Session Rewind (v0.1.60+)** section with rewind example
- Added `get_context_usage()` / context limit monitoring example
- Added **Revision History** table (v0.1.60, v0.1.59, v0.1.6)

---

## 6. Anthropic Claude Agent SDK (TypeScript) — `claude_agent_sdk_typescript_comprehensive_guide.md`

**Version**: Added `0.2.110 | Updated: April 2026` header

- Added **Revision History** table (v0.2.110, v0.1.30)

---

## 7. OpenAI Agents SDK (Python) — `openai_agents_sdk_comprehensive_guide.md`

**Version**: `0.6.1` → `0.14.1`

- Updated header
- Added **Breaking Changes in v0.14.x** table: `openai` v2 requirement, Python 3.9 dropped, sync auto-wrap, MCP errors as exceptions, `as_tool()` type narrowed
- Added **SandboxAgent (v0.14.0+)** section with `Manifest`, persistent workspace, and artifact access
- Added **WebSocket Streaming (v0.13.0+)** section with `responses_websocket_session()`
- Added **Revision History** table (v0.14.1, v0.14.0, v0.13.x, v0.6.1)

---

## 8. OpenAI Agents SDK (TypeScript) — `openai_agents_sdk_typescript_comprehensive_guide.md`

**Version**: `6.8.1` → `0.8.3` (corrected erroneous version number)

- Corrected header, updated `Last Updated` to April 2026
- Documented monorepo split: `openai-agents` core + `@openai/agents-extensions-*`
- Added **Realtime Agents with WebSocket Voice/Audio (v0.8.x)** section
- Added **Revision History** table (v0.8.3, v0.8.0, v0.3.2)

---

## 9. LangGraph (Python) — `langgraph_comprehensive_guide.md`

**Version**: `1.0.3` → `1.1.6`

- Updated header and version references
- Added **Type-Safe v2 API (v1.1.x)** section covering v2 streaming (`StreamPart`), v2 invoke (`GraphOutput`), and Pydantic/dataclass auto-coercion
- Added `create_react_agent` deprecation notice
- Added **Revision History** table (v1.1.6, v1.0.3)

---

## 10. LangGraph (TypeScript) — `langchain_langgraph_comprehensive_guide.md`

**Version**: `1.0.2` → `1.2.8`

- Updated header
- Added `createReactAgent` deprecation notice (moved to `@langgraphjs/toolkit`)
- Added **New Type Utilities (v1.2.x)** section: `ReducedValue` and `UntrackedValue`
- Added **Revision History** table (v1.2.8, v1.0.2)

---

## 11. LlamaIndex (Python)

**Version**: `0.14.8` → `0.14.20`

**Critical fixes** — all hard-removed classes replaced throughout:

| File | Changes |
|------|---------|
| `llamaindex_comprehensive_guide.md` | Sections 4.1–4.3 updated: `ReActAgent` → `AgentWorkflow`, `FunctionCallingAgent` → `AgentWorkflow`, `AgentRunner` → `AgentWorkflow` |
| `llamaindex_advanced_implementations.md` | All `ReActAgent` imports/usages replaced; `ObservableReActAgent` → `ObservableAgentWorkflow` |
| `llamaindex_recipes.md` | All `ReActAgent` occurrences replaced with `AgentWorkflow` |
| `GUIDE_INDEX.md` | Index entries updated: `ReActAgent` → `AgentWorkflow`, `OpenAIAgent` → `AgentWorkflow` |
| `llamaindex_observability_python.md` | Version header updated |
| `llamaindex_streaming_server_fastapi.md` | Version header updated |

Added **Revision History** table (v0.14.20, v0.14.8, v0.12.x).

---

## 12. Mistral Agents API — `mistral_agents_api_comprehensive_guide.md`

**Version**: `1.9.11` → `2.0.1`

- Added BREAKING change notice for v1.x → v2.x incompatibility
- Updated header
- Added **Revision History** table (v2.0.1, v1.9.11)

Additional Mistral files updated:
- `mistral_agents_api_advanced_python.md`: version header + BREAKING notice
- `mistral_streaming_server_fastapi.md`: version header updated

---

## 13. AG2 (AutoGen) — `ag2_comprehensive_guide.md`

**Version**: `0.3.2` → `0.11.5`

- Updated header, version, and date
- Added `autogen.beta` compatibility note
- Added **autogen.beta: Event-Driven Architecture (v0.11.x+)** section with `MemoryStream` pub/sub
- Added **A2A Protocol Support** section
- Added **AG2 CLI** section (`ag2 init`, `ag2 run`)
- Added **QuickResearchTool** section
- Added **Revision History** table (v0.11.5, v0.11.0, v0.3.2)

Additional AG2 files updated:
- `autogen_comprehensive_guide.md`: `0.10.0` → `0.11.5`
- `autogen_comprehensive_guide_updated.md`: `0.10.0` → `0.11.5`
- `ag2_recipes.md`, `ag2_production_guide.md`, `ag2_diagrams.md`: version + date

---

## 14. Haystack — `haystack_comprehensive_guide.md`

**Version**: `2.19.0` → `2.27.0`

- Updated header and "2025 → 2026" section titles
- Added Python 3.10+ requirement notice
- Added **SearchableToolset (v2.21.0+)** section
- Added **LLMRanker (v2.22.0+)** section
- Added **AgentToolInvoker (v2.23.0+)** section
- Added **Revision History** table (v2.27.0 through v2.19.0)

Additional Haystack files updated:
- `haystack_advanced_agents_python.md`: `2.19.0` → `2.27.0`
- `haystack_streaming_server_fastapi.md`: `2.19.0` → `2.27.0`
- `haystack_observability_python.md`: `2.19.0` → `2.27.0`

---

## 15. Google ADK (Python) — `google_adk_comprehensive_guide.md`

**Version**: `1.18.0` → `1.30.0`

- Updated header and installation
- Added **GraphAgent** section with graph-based multi-agent orchestration
- Added **Task API** section for structured task management
- Added **Session Rewind** section for replay/debugging
- Added **Revision History** table (v1.30.0 through v1.18.0)

Additional Google ADK Python files updated:
- `google_adk_advanced_python.md`: `1.18.0` → `1.30.0`
- `python/GUIDE_INDEX.md`: version + date

---

## 16. Google ADK (Go) — `google_adk_go_comprehensive_guide.md`

**Version**: `0.1.0` → `1.0.0 GA`

- Added GA release banner
- Added **OpenTelemetry Integration** section
- Added **YAML Agent Configuration** section
- Added **Plugin System** section
- Updated `Last Updated` date
- Added **Revision History** table (v1.0.0 GA, v0.1.0)

Additional Go ADK files updated:
- `GUIDE_INDEX.md`: version → `1.0.0 GA`
- `google_adk_go_recipes.md`, `google_adk_go_production_guide.md`: date updated

---

## 17. Semantic Kernel (.NET) — `semantic_kernel_comprehensive_dotnet.md`

**Version**: `1.38.0` → `1.74.0`

- Updated version references throughout
- Noted `Planners.Handlebars` removal (breaking)
- Added **Revision History** table (v1.74.0, v1.67.1)

Additional .NET files updated (7 files):
- `semantic_kernel_recipes_dotnet.md`, `semantic_kernel_production_dotnet.md`, `semantic_kernel_diagrams_dotnet.md`, `GUIDE_INDEX.md`: version + date

---

## 18. Semantic Kernel (Python) — `semantic_kernel_comprehensive_python.md`

**Version**: `1.38.0` → `1.41.2`

- Added **MCP Server/Client Support (v1.41.x)** section
- Added **A2A Protocol Support (v1.41.x)** section
- Added Python 3.10+ requirement note
- Added **Revision History** table (v1.41.2, v1.38.0)

Additional Python SK files updated (6 files):
- `semantic_kernel_production_python.md`, `semantic_kernel_recipes_python.md`, `semantic_kernel_diagrams_python.md`, `GUIDE_INDEX.md`, `semantic_kernel_streaming_server_fastapi.md`: version + date

Top-level `semantic_kernel_comprehensive_guide.md`: `1.38.0` → `SK Python 1.41.2 / .NET 1.74.0`

---

## 19. Microsoft Agent Framework — `microsoft_agent_framework_comprehensive_guide.md`

**Version**: `1.0 Preview` → `1.0 GA`

- Updated header to GA 1.0 (April 3, 2026)
- Removed all `--prerelease` / `--pre` install flags
- Added `ChatClientAgentOptions.Instructions` removal note
- Added **Revision History** table (v1.0 GA, v1.0 Preview)

---

## 20. Amazon Bedrock — `bedrock_agents_comprehensive_guide.md`

- Added version header: **Strands SDK 1.35.0 (April 2026)**
- Added `Last Updated: April 2026`

Additional files:
- `bedrock_strands_sdk_guide.md`: version `1.35.0` + date added
- `bedrock_agentcore_comprehensive_guide.md`: date added

---

## 21. Site Index Files

### `index.md` (root)
- Updated to April 17, 2026
- "What's New" section rewritten for April 2026 release cycle
- All LangGraph references updated to v1.1.6
- File count updated: 160+ → 219+

### `docs/guides.md`
- Updated to April 17, 2026
- All framework version numbers updated throughout
- File count updated: 160+ → 219+

---

## Files Modified (55 total)

| File | Change Type |
|------|-------------|
| `versions.json` | 3 version bumps, notes updated |
| `index.md` | Dates, versions, What's New section |
| `docs/guides.md` | All version references updated |
| `PydanticAI_Guide/pydantic_ai_comprehensive_guide.md` | v1.84.0, new features, revision history |
| `SmolAgents_Guide/smolagents_comprehensive_guide.md` | v1.24.0, HfApiModel note, revision history |
| `CrewAI_Guide/crewai_comprehensive_guide.md` | v1.14.0, CheckpointConfig, providers, revision history |
| `CrewAI_Guide/crewai_flows_guide.md` | Version references |
| `Anthropic_Claude_Agent_SDK_Guide/anthropic_claude_agent_sdk_comprehensive_guide.md` | v0.1.60, thinking config, checkpointing, revision history |
| `Anthropic_Claude_Agent_SDK_TypeScript_Guide/claude_agent_sdk_typescript_comprehensive_guide.md` | v0.2.110 header, revision history |
| `OpenAI_Agents_SDK_Guides/openai_agents_sdk_comprehensive_guide.md` | v0.14.1, SandboxAgent, WebSocket, revision history |
| `OpenAI_Agents_SDK_TypeScript_Guide/openai_agents_sdk_typescript_comprehensive_guide.md` | v0.8.3 corrected, Realtime Agents, revision history |
| `LangGraph_Guide/python/langgraph_comprehensive_guide.md` | v1.1.6, v2 API, revision history |
| `LangGraph_Guide/typescript/langchain_langgraph_comprehensive_guide.md` | v1.2.8, new types, revision history |
| `LlamaIndex_Guide/python/llamaindex_comprehensive_guide.md` | v0.14.20, AgentWorkflow, revision history |
| `LlamaIndex_Guide/python/llamaindex_advanced_implementations.md` | ReActAgent → AgentWorkflow |
| `LlamaIndex_Guide/python/llamaindex_recipes.md` | ReActAgent → AgentWorkflow |
| `LlamaIndex_Guide/python/GUIDE_INDEX.md` | Index entries updated |
| `LlamaIndex_Guide/python/llamaindex_observability_python.md` | Version header |
| `LlamaIndex_Guide/python/llamaindex_streaming_server_fastapi.md` | Version header |
| `Mistral_Agents_API_Guide/mistral_agents_api_comprehensive_guide.md` | v2.0.1, BREAKING notice, revision history |
| `Mistral_Agents_API_Guide/mistral_agents_api_advanced_python.md` | v2.0.1, BREAKING notice |
| `Mistral_Agents_API_Guide/mistral_streaming_server_fastapi.md` | v2.0.1 header |
| `AG2_Guide/ag2_comprehensive_guide.md` | v0.11.5, autogen.beta, A2A, CLI, revision history |
| `AG2_Guide/ag2_diagrams.md` | v0.11.5, date |
| `AG2_Guide/ag2_recipes.md` | v0.11.5, date |
| `AG2_Guide/ag2_production_guide.md` | v0.11.5, date |
| `AutoGen_Guide/python/autogen_comprehensive_guide.md` | v0.11.5, revision history |
| `AutoGen_Guide/python/autogen_comprehensive_guide_updated.md` | v0.11.5, revision history |
| `Haystack_Guide/haystack_comprehensive_guide.md` | v2.27.0, SearchableToolset, LLMRanker, AgentToolInvoker, revision history |
| `Haystack_Guide/haystack_advanced_agents_python.md` | v2.27.0 header |
| `Haystack_Guide/haystack_streaming_server_fastapi.md` | v2.27.0 header |
| `Haystack_Guide/haystack_observability_python.md` | v2.27.0 header |
| `Google_ADK_Guide/python/google_adk_comprehensive_guide.md` | v1.30.0, GraphAgent, Task API, Session Rewind, revision history |
| `Google_ADK_Guide/python/google_adk_advanced_python.md` | v1.30.0 header |
| `Google_ADK_Guide/python/GUIDE_INDEX.md` | v1.30.0, date |
| `Google_ADK_Guide/go/google_adk_go_comprehensive_guide.md` | v1.0.0 GA, OTel, YAML, plugins, revision history |
| `Google_ADK_Guide/go/GUIDE_INDEX.md` | v1.0.0 GA |
| `Google_ADK_Guide/go/google_adk_go_recipes.md` | Date |
| `Google_ADK_Guide/go/google_adk_go_production_guide.md` | Date |
| `Semantic_Kernel_Guide/semantic_kernel_comprehensive_guide.md` | SK Python 1.41.2 / .NET 1.74.0, date |
| `Semantic_Kernel_Guide/dotnet/semantic_kernel_comprehensive_dotnet.md` | v1.74.0, revision history |
| `Semantic_Kernel_Guide/dotnet/semantic_kernel_recipes_dotnet.md` | v1.74.0, date |
| `Semantic_Kernel_Guide/dotnet/semantic_kernel_production_dotnet.md` | v1.74.0, date |
| `Semantic_Kernel_Guide/dotnet/semantic_kernel_diagrams_dotnet.md` | Date |
| `Semantic_Kernel_Guide/dotnet/GUIDE_INDEX.md` | Date |
| `Semantic_Kernel_Guide/python/semantic_kernel_comprehensive_python.md` | v1.41.2, MCP, A2A, revision history |
| `Semantic_Kernel_Guide/python/semantic_kernel_production_python.md` | v1.41.2, date |
| `Semantic_Kernel_Guide/python/semantic_kernel_recipes_python.md` | v1.41.2, date |
| `Semantic_Kernel_Guide/python/semantic_kernel_diagrams_python.md` | Date |
| `Semantic_Kernel_Guide/python/GUIDE_INDEX.md` | v1.41.2, date |
| `Semantic_Kernel_Guide/python/semantic_kernel_streaming_server_fastapi.md` | v1.41.2 header |
| `Microsoft_Agent_Framework_Guide/microsoft_agent_framework_comprehensive_guide.md` | GA 1.0, `--prerelease` removed, revision history |
| `Amazon_Bedrock_Agents_Guide/bedrock_agents_comprehensive_guide.md` | Strands 1.35.0, date |
| `Amazon_Bedrock_Agents_Guide/bedrock_strands_sdk_guide.md` | v1.35.0, date |
| `Amazon_Bedrock_Agents_Guide/bedrock_agentcore_comprehensive_guide.md` | Date |

---

## Cross-Cutting Changes

### Revision History Added
All 19 comprehensive framework guides now include a **Revision History** table documenting version changes with dates and descriptions.

### Code Example Fixes
| Issue | Files Fixed | Change |
|-------|------------|--------|
| `ReActAgent` (ImportError in v0.14.x) | 4 LlamaIndex files | → `AgentWorkflow` |
| `FunctionCallingAgent` (ImportError) | 2 LlamaIndex files | → `AgentWorkflow` |
| `AgentRunner` (ImportError) | 1 LlamaIndex file | → `AgentWorkflow` |
| `HfApiModel` (deprecated) | SmolAgents guide | Added deprecation note + `InferenceClientModel` |
| `CodeInterpreterTool` (removed) | CrewAI guide | Added ImportError warning + custom tool alternative |
| `crewai==1.4.1` (pinned old version) | CrewAI guide | → `crewai==1.14.0` |
| `--prerelease` flags | Microsoft AF guide | Removed (GA release) |
| `semantic-kernel==1.38.0` pin | SK Python guide | → `>=1.41.2` |

---

*Report generated: April 17, 2026*
*Previous report: [update_report_april_2026.md](update_report_april_2026.md)*

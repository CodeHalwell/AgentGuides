# Agent Guides Update Report — April 16, 2026

**Summary**: Comprehensive update of all agent framework guides from November 2025 to April 2026. Covers version upgrades, breaking API changes, new features, deprecations, and the addition of a new Google ADK TypeScript guide.

---

## 1. `versions.json`

Complete rewrite updating all framework version records:

| Framework | Before | After |
|-----------|--------|-------|
| OpenAI Agents SDK (Python) | 0.6.1 | 0.14.x |
| OpenAI Agents SDK (TypeScript) | 0.3.2 | 0.8.3 |
| PydanticAI | 1.20.0 | 1.83.0 |
| CrewAI | 1.5.0 | 1.14.0 |
| AG2 | 0.3.2 | 0.11.5 |
| LangGraph (Python) | 1.0.3 | 1.1.6 |
| LangGraph (TypeScript) | 1.0.2 | 1.2.8 |
| SmolAgents | 1.23.0 | 1.24.0 |
| Haystack | 2.20.0 | 2.27.0 |
| LlamaIndex (Python) | 0.12.x | 0.14.20 |
| Microsoft Agent Framework | 1.0 Preview | 1.0 GA |
| Mistral Agents API SDK | 1.9.11 | 2.0.1 |
| Anthropic Python | 0.1.6 | 0.1.59 |
| Anthropic TypeScript | 0.1.30 | 0.2.110 |
| Google ADK (Python) | 1.18.0 | 1.30.0 |
| Google ADK (Go) | 0.1.0 | 1.0.0 GA |
| Amazon Bedrock Strands | ~1.0 | 1.35.0 |
| Semantic Kernel (.NET) | 1.67.1 | 1.74.0 |
| Semantic Kernel (Python) | 1.38.0 | 1.41.2 |

**New entry added**: `google-adk-typescript` (announced December 2025)

---

## 2. OpenAI Agents SDK (Python) — `OpenAI_Agents_SDK_Guides/README.md`

**Breaking changes documented**:
- `openai` v2 is now a hard requirement (v1.x raises `ImportError`)
- Python 3.9 support dropped; minimum is now 3.10
- Sync tool functions are now automatically wrapped with `asyncio.to_thread`
- MCP tool errors now propagate as exceptions rather than strings
- `as_tool()` return type narrowed to `FunctionTool`

**New features added**:
- **Sandbox Agents** (`SandboxAgent`, `Manifest`) — isolated execution with configurable resource limits
- **WebSocket transport** (`responses_websocket_session()`) for real-time streaming

---

## 3. OpenAI Agents SDK (TypeScript) — `OpenAI_Agents_SDK_TypeScript_Guide/README.md`

**Breaking changes documented**:
- Monorepo package split: `openai-agents` core + `@openai/agents-extensions-*` sub-packages
- `aisdk()` import path changed to `@openai/agents-extensions-vercel-ai-sdk`

**New features added**:
- Realtime Agents with WebSocket voice/audio
- Sessions API for multi-turn persistence
- Zod schema validation for tool inputs
- Built-in tracing (previously external-only)
- Note: Sandbox Agents not yet available in TypeScript

---

## 4. PydanticAI — `PydanticAI_Guide/README.md`

**Breaking changes (hard-removed in v1.83.0)**:
- 9 API renames, all hard-removed (no deprecation warnings; raises `AttributeError`):

| Old name | New name |
|----------|----------|
| `result_type` | `output_type` |
| `result_retries` | `output_retries` |
| `result_validators` | `output_validators` |
| `RunResult.data` | `RunResult.output` |
| `StreamedRunResult.get()` | `StreamedRunResult.get_output()` |
| `capture_run_messages` | `capture_output_messages` |
| `last_run_messages` | `last_output_messages` |
| `agent.run_result_type` | `agent.output_type` |
| `FunctionModel.result_schema` | `FunctionModel.output_schema` |

**New features**:
- `EvaluationReport` API for LLM-based eval
- `pydantic-graph` expansion: branching/looping graph workflows
- `defer_loading` for lazy model init
- `ThreadExecutor` for sync-in-async tool calls
- Smart instruction caching
- `CaseLifecycle` hooks for state machine patterns
- Local `WebFetch` tool

---

## 5. CrewAI — `CrewAI_Guide/README.md`

**Breaking changes**:
- `CodeInterpreterTool` hard-removed (raises `ImportError`)
- `CrewStructuredTool` moved to internal API
- `stop` parameter no longer sent to GPT-5/o-series models
- v1.10.0 was yanked from PyPI; use 1.10.1+

**New features**:
- **Checkpoint System**: `CheckpointConfig` + `SqliteProvider` for resumable crew execution
- CLI: `crewai checkpoint list` / `crewai checkpoint info`
- Structured Pydantic outputs via `response_format`
- Before/after tool call hooks
- Native GPT-5/o-series vision support
- SSRF and path traversal protections in RAG tools
- Enterprise RBAC and SSO documentation
- MCP custom server support

---

## 6. AG2 — `AG2_Guide/README.md`

**Version**: 0.3.2 → 0.11.5

**New features**:
- `autogen.beta`: MemoryStream pub/sub event-driven architecture
- A2A (Agent-to-Agent) protocol support
- `A2UIAgent` for UI interaction
- AG2 CLI (`ag2 init`, `ag2 run`)
- `RemyxCodeExecutor` for containerised code execution
- `GroupToolExecutor` async handler
- Multi-provider support (Bedrock, Vertex, local models)
- v1.0 roadmap and deprecation timeline documented

---

## 7. LangGraph (Python) — `LangGraph_Guide/python/README.md`

**Breaking changes**:
- Python 3.9 dropped (minimum 3.10)
- `langgraph.prebuilt.create_react_agent` deprecated; use `langchain.agents.create_agent`
- `langgraph-prebuilt==1.0.9` introduces `ImportError` for `ServerInfo` when paired with older versions

**New features (v1.1.x)**:
- Type-safe streaming v2 API (opt-in with `version="v2"`): unified `StreamPart` output
- Type-safe invoke v2 API: returns `GraphOutput` with `.value` and `.interrupts`
- Pydantic/dataclass auto-coercion on `invoke()`
- Python 3.14 support
- Fixed time-travel bugs with interrupts and subgraphs

---

## 8. LangGraph (TypeScript) — `LangGraph_Guide/typescript/README.md`

**Breaking changes**:
- `createReactAgent` moved to `@langgraphjs/toolkit` (deprecated in `@langchain/langgraph/prebuilt`)

**New features**:
- Standard JSON Schema support (Zod 4, Valibot, ArkType compatible)
- `ReducedValue` type: fields with separate input/output schemas
- `UntrackedValue` type: transient state that is never checkpointed

---

## 9. SmolAgents — `SmolAgents_Guide/README.md`

**Deprecations**:
- `HfApiModel` deprecated; migrate to `InferenceClientModel`
- Python 3.9 no longer supported

**New features (v1.24.0)**:
- Backward compatibility shim for `HfApiModel`
- Expanded model compatibility for `gpt-5.2*` and newer OpenAI families
- `token_counts` tracking fix for managed agents called multiple times
- Vision model support for web browsing agents

---

## 10. Haystack — `Haystack_Guide/README.md`

**Version**: 2.20.0 → 2.27.0

**Breaking changes**:
- Python 3.9 dropped (minimum 3.10)
- `ChatMessage` internal structure refactored

**New features (v2.21–v2.27)**:
- `SearchableToolset` for dynamic tool sets from document stores
- `LLMRanker` built-in ranker component
- `AgentToolInvoker` for declarative tool calling
- Structured component outputs with Pydantic
- Native OpenAI Responses API support
- `GreedyVariadicRouterComponent` for fan-out workflows
- Incremental document store updates
- Per-message cost tracking
- 42 new integrations

---

## 11. LlamaIndex (Python) — `LlamaIndex_Guide/python/README.md`

**Hard removals (raises `ImportError`)**:
- `llama_index.core.agent.ReActAgent` — use `AgentWorkflow`
- `llama_index.core.agent.OpenAIAgent` — use `AgentWorkflow`
- `llama_index.core.agent.StructuredPlanningAgent` — use `AgentWorkflow`
- `llama_index.agent.react.ReActAgent` — same as above

**Breaking changes**:
- Python 3.9 dropped
- TypeScript package split: `@llamaindex/workflow` is now separate
- `asyncio_module` parameter deprecated

**New features**:
- `AgentWorkflow` is the primary recommended agent pattern
- LlamaSheets integration
- LiteParse document parser
- Agent Client Protocol for cross-framework communication

---

## 12. Microsoft Agent Framework — `Microsoft_Agent_Framework_Guide/README.md`

**Status changed**: Public Preview → **GA 1.0** (April 3–7, 2026)

**Breaking changes (Preview → GA)**:
- `ChatClientAgentOptions.Instructions` property removed
- New package name scheme and version floor requirements
- `--prerelease` flags removed from all install commands

**New in GA 1.0**:
- First-party connectors: Azure OpenAI, OpenAI, Anthropic, Bedrock, Gemini, Ollama
- Azure App Service deployment support
- Long-Term Support (LTS) designation

---

## 13. Mistral Agents API — `Mistral_Agents_API_Guide/README.md`

**Version**: SDK 1.9.11 → **2.0.1** (March 12, 2026)

**Breaking changes (v1 → v2 SDK)**:
- Install: `mistralai` v2.x (`pip install mistralai>=2.0.0`)
- Stateful conversation API redesigned
- TypeScript SDK: ESM-only module system
- TypeScript SDK: requires Zod v4

---

## 14. Anthropic Claude Agent SDK (Python) — `Anthropic_Claude_Agent_SDK_Guide/README.md`

**Version**: 0.1.6 → 0.1.59

**Breaking changes**:
- Import path: `claude_code_sdk` → `claude_agent_sdk`
- Class: `ClaudeCodeOptions` → `ClaudeAgentOptions`
- `sandbox.failIfUnavailable` default changed

**New features**:
- `get_context_usage()` for token tracking
- `typing.Annotated` for per-parameter descriptions
- Structured outputs
- `reloadPlugins()` for dynamic plugin management
- Bundled CLI
- Fallback model handling

---

## 15. Anthropic Claude Agent SDK (TypeScript) — `Anthropic_Claude_Agent_SDK_TypeScript_Guide/README.md`

**Version**: 0.1.30 → 0.2.110

**Breaking changes**:
- Package renamed: `@anthropic-ai/claude-code` → `@anthropic-ai/claude-agent-sdk`
- `sandbox.failIfUnavailable` default changed

**New features**:
- Structured outputs with Zod
- MCP integration
- Multibyte character fix
- MCP cleanup fix

---

## 16. Google ADK (Python) — `Google_ADK_Guide/python/README.md`

**Version**: 1.18.0 → 1.30.0

**New features**:
- Graph-based agent workflows (`GraphAgent`)
- Task API for structured task management
- `AgentEngineSandboxCodeExecutor`
- Session rewind for replay/debugging
- A2A 1.0 specification compliance
- Web UI updates

**Deprecations**:
- `MCPToolset` legacy API deprecated; async-first migration recommended

---

## 17. Google ADK (Go) — `Google_ADK_Guide/go/README.md`

**Version**: 0.1.0 → **1.0.0 GA** (April 8, 2026)

**New in 1.0.0**:
- OpenTelemetry integration
- A2A 1.0 full support
- YAML-based agent configuration
- Plugin system
- Tool confirmation hooks

**Breaking changes from 0.x**:
- Agent constructor signature changed
- Tool registration API updated

---

## 18. Google ADK (TypeScript) — `Google_ADK_Guide/typescript/README.md` ⭐ NEW

**Entirely new guide** for TypeScript ADK support (announced December 2025).

**Covers**:
- Installation (`@google/adk`)
- `LlmAgent` and `FunctionTool` with Zod schemas
- Multi-agent systems with `AgentTool`
- Sessions (`InMemorySessionService`)
- Streaming with async iteration
- MCP integration (`MCPToolset`)
- A2A protocol (`A2AServer`)
- Development tools (`adk web`, `adk run`, `adk deploy`)
- Production deployment (Cloud Run, Vertex AI Agent Engine)
- Python vs TypeScript comparison table

---

## 19. Semantic Kernel (.NET) — `Semantic_Kernel_Guide/dotnet/README.md`

**Version**: 1.67.1 → 1.74.0

**Breaking changes**:
- Handlebars planner packages removed
- OpenAI planner packages deprecated (migration to built-in planners)

**New features**:
- Server URL validation
- Text Search LINQ provider
- Plugin graduation from alpha to preview
- CVE-2026-26127 security fix

**Strategic note**: Microsoft Agent Framework 1.0 GA is now recommended for new multi-agent projects.

---

## 20. Semantic Kernel (Python) — `Semantic_Kernel_Guide/python/README.md`

**Version**: 1.38.0 → 1.41.2

**New features**:
- Full MCP server/client support
- A2A protocol support
- Oracle database connector
- Google GenAI SDK migration from `google-generativeai`

**Changed**:
- Python minimum version: 3.9+ → 3.10+

---

## 21. Amazon Bedrock Agents — `Amazon_Bedrock_Agents_Guide/README.md`

**Version**: Strands SDK ~1.0 → 1.35.0 | TypeScript SDK 1.0.0-rc.3

**New features**:
- Multi-agent graph workflows
- Interrupt propagation between agents
- `AgentCoreMemorySessionManager`
- Steering Hooks for execution control
- Structured output support
- Prompt caching
- Llama 4 and Nova Forge model support

**Behavioral changes**:
- `retry_strategy=None` now disables retries entirely
- A2A server `agentCard` URL format changed

---

## Cross-Cutting Changes

### Python 3.9 End-of-Life
The following frameworks dropped Python 3.9 support in this update cycle:
- LangGraph (Python)
- LlamaIndex
- Haystack
- SmolAgents
- Semantic Kernel (Python)

Minimum Python version across all Python-based guides is now **3.10**.

### A2A (Agent-to-Agent) Protocol
A2A 1.0 is now supported across: Google ADK (Python, Go, TypeScript), AG2, Semantic Kernel (Python), Amazon Bedrock Strands.

### MCP (Model Context Protocol)
MCP integration is documented across: CrewAI, LangGraph, OpenAI Agents SDK, SmolAgents, Semantic Kernel, Google ADK, Anthropic Claude SDK, Haystack.

---

## Files Modified

| File | Action |
|------|--------|
| `versions.json` | Updated all versions, added `google-adk-typescript` |
| `README.md` (root) | Added Google ADK TypeScript entry, revision history |
| `OpenAI_Agents_SDK_Guides/README.md` | Breaking changes, Sandbox Agents, WebSocket transport |
| `OpenAI_Agents_SDK_TypeScript_Guide/README.md` | Monorepo split, Realtime Agents, Sessions API |
| `PydanticAI_Guide/README.md` | `result_*` → `output_*` hard renames, new features |
| `CrewAI_Guide/README.md` | CodeInterpreterTool removed, CheckpointConfig |
| `AG2_Guide/README.md` | v0.11.5, autogen.beta, A2A, CLI |
| `LangGraph_Guide/README.md` | Updated versions, revision history |
| `LangGraph_Guide/python/README.md` | v1.1.6, Python 3.9 drop, v2 API, create_react_agent deprecation |
| `LangGraph_Guide/typescript/README.md` | v1.2.8, createReactAgent moved, ReducedValue/UntrackedValue |
| `SmolAgents_Guide/README.md` | v1.24.0, HfApiModel deprecation, vision support |
| `Haystack_Guide/README.md` | v2.27.0, SearchableToolset, LLMRanker, Python 3.9 drop |
| `LlamaIndex_Guide/README.md` (root) | Hard removals, AgentWorkflow, Python 3.9 drop |
| `LlamaIndex_Guide/python/README.md` | Revision history added |
| `LlamaIndex_Guide/typescript/README.md` | Package restructuring, revision history |
| `Microsoft_Agent_Framework_Guide/README.md` | GA 1.0, breaking changes, connectors |
| `Microsoft_Agent_Framework_Guide/dotnet/README.md` | GA 1.0, revision history |
| `Microsoft_Agent_Framework_Guide/python/README.md` | GA 1.0, revision history |
| `Mistral_Agents_API_Guide/README.md` | SDK v2.0.1, breaking changes |
| `Anthropic_Claude_Agent_SDK_Guide/README.md` | v0.1.59, import path changes |
| `Anthropic_Claude_Agent_SDK_TypeScript_Guide/README.md` | v0.2.110, package rename |
| `Amazon_Bedrock_Agents_Guide/README.md` | Strands 1.35.0, graph workflows |
| `Google_ADK_Guide/README.md` | v1.30.0 / v1.0.0 GA, TypeScript announcement |
| `Google_ADK_Guide/python/README.md` | v1.30.0, Task API, session rewind |
| `Google_ADK_Guide/go/README.md` | v1.0.0 GA, OTel, YAML config |
| `Google_ADK_Guide/typescript/README.md` | **NEW FILE** — TypeScript ADK guide |
| `Semantic_Kernel_Guide/README.md` | Updated versions, revision history |
| `Semantic_Kernel_Guide/dotnet/README.md` | v1.74.0, planner removals |
| `Semantic_Kernel_Guide/python/README.md` | v1.41.2, MCP, A2A, Oracle connector |

---

*Report generated: April 16, 2026*

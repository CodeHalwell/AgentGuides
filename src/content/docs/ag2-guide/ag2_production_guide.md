---
title: "AG2 Production Guide"
description: "Running AG2 1.x in production — pinning, testing with TestConfig, durable hubs, serving over A2A, observability, security and scaling."
framework: ag2
---

> **Verified against `ag2` %%version=ag2%%** (checked %%versions-checked%%).

## Pin the right distribution

`ag2` changed API completely at 1.0. Pin a range that matches the code you wrote:

```text
# requirements.txt — AG2 1.x code (import ag2)
ag2[openai,mcp]>=%%version=ag2%%,<2

# AG2 Classic code (import autogen) — maintenance mode
# autogen[openai]
```

Never leave `ag2` unpinned in a Classic codebase: a fresh install would pull 1.x and every `import autogen` would fail.

## Test without a model

Use `ag2.testing.TestConfig` to script model outputs (strings and `ToolCallEvent`s). Tests then exercise your tools, schemas and control flow deterministically and offline — see [Testing](./ag2_comprehensive_guide/#testing). Keep a small separate suite that hits the real provider, run on a schedule rather than on every commit.

## Deployment

### Container

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "app"]
```

### Serving an agent

- **A2A** — `A2AServer(agent).build_jsonrpc(...)` returns an ASGI app; run it under uvicorn or any ASGI server. REST and gRPC transports can share one `task_store=`; for several replicas, back it with a shared store rather than the default in-memory one.
- **MCP** — `ag2.mcp.MCPServer(agent)` exposes an `Agent` as an MCP server for other clients.
- **AG-UI / ACP** — adapters in `ag2.ag_ui` and `ag2.acp` (extras `ag-ui`, `acp`) connect agents to front-ends and editor clients.

## State and durability

- A Network `Hub` persists its registry and write-ahead log through a `KnowledgeStore`. `MemoryKnowledgeStore` is for tests; use `DiskKnowledgeStore`, `SqliteKnowledgeStore` or `RedisKnowledgeStore` (extra `redis`) in production so channels can be restored with `Hub.hydrate()` after a restart.
- `TransitionGraph`s are plain data (`to_dict()`), so workflow channels survive restarts without re-registering Python callables.
- For long single-agent sessions, attach `knowledge=KnowledgeConfig(...)` with a compaction strategy (`ag2.compact.SummarizeCompact`) to cap history growth.

## Resilience

- Provider configs accept `timeout=` and `max_retries=` (OpenAI-family) — set both explicitly.
- Tool exceptions propagate to the `ask()` call; catch them at the boundary, or convert expected failures into returned error strings so the model can recover.
- Bound multi-agent runs: `TransitionGraph(..., max_turns=N)` for workflows, and an explicit `close()` plus a timeout for `conversation` and `discussion` channels, which never end on their own.
- A2A clients reconnect on dropped streams (`max_reconnects`, `reconnect_backoff` on `A2AConfig`) and raise `A2AReconnectError` when the budget is spent.

## Security

- **Code execution** — prefer `SandboxCodeTool` / container environments over `LocalEnvironment` and `ShellTool` for anything model-generated.
- **MCP** — restrict exposed tools with `allowed_tools=` / `blocked_tools=`; prefer `MCPToolkit` (client-side) when credentials must stay inside your infrastructure.
- **A2A push notifications** — if you enable them (`push_config_store=`), validate client-supplied webhook URLs in your `push_sender` so they cannot reach internal addresses (SSRF). Sign agent cards with `card_signer=` where clients need to verify them; clients check with `card_signature_verifier=` on `A2AConfig`.
- **Keys** — config classes read provider keys from the environment; keep them in a secrets manager, not in code.

## Observability

- Subscribe to typed events on a `MemoryStream` (tool calls, model responses, human-input requests, compaction) for audit logs and UIs.
- `observers=` and `middleware=` on `Agent` attach cross-cutting behaviour — metrics, policies, redaction — without touching agent code.
- Install the `tracing` extra for OpenTelemetry spans and `metrics` for metrics export.
- The Hub's audit trail records agent registration, channel lifecycle and expectation violations.

## Scaling

- Agents are cheap async objects; run many concurrent `ask()` calls in one process before reaching for more processes.
- For horizontal scale, put stateless agent services behind a load balancer and keep conversation state in the hub's store or your own database.
- Watch token usage per turn (`ag2.usage`) and set provider-side budgets.

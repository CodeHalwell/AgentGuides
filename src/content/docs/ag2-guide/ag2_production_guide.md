---
title: "AG2 Production Guide"
description: "Running AG2 1.x in production — pinning, testing with TestConfig, durable hubs, serving over A2A, observability, security and scaling."
framework: "ag2"
language: "python"
sidebar:
  label: "Production"
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

Never leave `ag2` unpinned in a Classic codebase: a fresh install would pull 1.x and every `import autogen` would fail. Running Classic in production is covered in the [AG2 Classic production guide](/ag2-guide/classic/ag2_classic_production_guide/).

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

- **A2A** — `A2AServer(agent).build_jsonrpc(...)` returns an ASGI app; run it under uvicorn or any ASGI server. REST and gRPC transports can share one `task_store=`. Since 1.2.0 `A2AServer` can run on several replicas: pass a versioned shared task store together with an `event_stream=` that carries task events between replicas (see the upstream A2A server docs).
- **MCP** — `ag2.mcp.MCPServer(agent)` exposes an `Agent` as an MCP server: `await server.run_stdio()` for a subprocess client, or run the server object itself under uvicorn for streamable HTTP at `/mcp`. See [Serving an agent as an MCP server](/ag2-guide/ag2_comprehensive_guide/#serving-an-agent-as-an-mcp-server).
- **AG-UI / ACP** — adapters in `ag2.ag_ui` and `ag2.acp` (extras `ag-ui`, `acp`) connect agents to front-ends and editor clients.

## State and durability

- A Network `Hub` persists its registry and write-ahead log through a `KnowledgeStore`. `MemoryKnowledgeStore` is for tests; use `DiskKnowledgeStore`, `SqliteKnowledgeStore` or `RedisKnowledgeStore` (extra `redis`) in production so channels can be restored with `Hub.hydrate()` after a restart.
- `TransitionGraph`s are plain data (`to_dict()`), so workflow channels survive restarts without re-registering Python callables.
- For long single-agent sessions, attach `knowledge=KnowledgeConfig(...)` with a compaction strategy (`TailWindowCompact` or `SummarizeCompact` from `ag2.compact`) to cap history growth. See [Knowledge, memory and compaction](/ag2-guide/ag2_comprehensive_guide/#knowledge-memory-and-compaction).

## Resilience

- Provider configs accept `timeout=` and `max_retries=` (OpenAI-family) — set both explicitly.
- A tool that raises does not fail `ask()`. AG2 publishes a `ToolErrorEvent` and hands it to the model as the tool result, and that text includes the Python traceback. So catch expected failures inside the tool and return a short message the model can act on, and keep secrets out of exception messages. `ag2.testing.TestConfig` behaves differently by default: it re-raises the tool's exception so a test fails loudly. Pass `raise_tool_errors=False` to model what a real provider does.
- Bound multi-agent runs: `TransitionGraph(..., max_turns=N)` for workflows, and an explicit `close()` plus a timeout for `conversation` and `discussion` channels, which never end on their own.
- A2A clients reconnect on dropped streams (`max_reconnects`, `reconnect_backoff` on `A2AConfig`) and raise `ag2.a2a.errors.A2AReconnectError` when the budget is spent.

## Security

- **Code execution**: run model-written code through `SandboxCodeTool` / `SandboxShellTool` with a `DockerEnvironment` or `DaytonaEnvironment`. `LocalEnvironment` is a plain subprocess with no isolation, so keep it for trusted code. `CodeExecutionTool` and `ShellTool` run in the model provider's sandbox, not on your infrastructure; check that this is acceptable for your data. See [Code execution](/ag2-guide/ag2_comprehensive_guide/#code-execution).
- **MCP** — restrict exposed tools with `allowed_tools=` / `blocked_tools=`; prefer `MCPToolkit` (client-side) when credentials must stay inside your infrastructure.
- **A2A push notifications** — if you enable them (`push_config_store=`), pass `push_url_validator=` (for example `a2a.utils.push_url_validator.validate_push_notification_url`, which allows only public http(s) hosts) so client-supplied webhook URLs cannot reach internal addresses (SSRF); available since 1.2.0. `validate_input_modes=True` rejects message parts whose media type the card does not declare. Sign agent cards with `card_signer=` where clients need to verify them; clients check with `card_signature_verifier=` on `A2AConfig`.
- **Keys** — config classes read provider keys from the environment; keep them in a secrets manager, not in code.

## Observability

`TelemetryMiddleware` (extra `tracing`) emits OpenTelemetry spans for agent turns, LLM calls, tool executions and human input, following the GenAI semantic conventions. Each reply carries a usage report:

```python
import asyncio

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from ag2 import Agent
from ag2.config import OpenAIConfig
from ag2.middleware import TelemetryMiddleware

provider = TracerProvider(resource=Resource.create({"service.name": "support-bot"}))
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))   # use an OTLP exporter in production

agent = Agent(
    "support",
    prompt="You answer support questions.",
    config=OpenAIConfig(model="%%model=openai-small%%"),
    middleware=[TelemetryMiddleware(tracer_provider=provider, capture_content=False)],
)


async def main() -> None:
    reply = await agent.ask("How do I reset my password?")
    report = await reply.usage()          # ag2.usage.UsageReport for the whole run
    print(report.total, dict(report.by_model))


asyncio.run(main())
```

- `capture_content=False` keeps prompts, completions and tool arguments out of spans. The default (`True`) records them.
- Subscribe to typed events on a `MemoryStream` (tool calls, model responses, human-input requests, compaction) for audit logs and UIs.
- `observers=[TokenMonitor(...), LoopDetector(...)]` alert on runaway token use and repeated tool calls. `MetricsMiddleware` (extra `metrics`) exports metrics. See [Middleware and observers](/ag2-guide/ag2_comprehensive_guide/#middleware-and-observers).
- The Hub's audit trail records agent registration, channel lifecycle and expectation violations.

## Scaling

- Agents are cheap async objects; run many concurrent `ask()` calls in one process before reaching for more processes.
- For horizontal scale, put stateless agent services behind a load balancer and keep conversation state in the hub's store or your own database.
- Watch token usage per run (`await reply.usage()`) and set provider-side budgets.

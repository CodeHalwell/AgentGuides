---
title: "LlamaIndex Observability & Monitoring (Python)"
description: "Tracing, token accounting, metrics and logging for LlamaIndex using the instrumentation module, OpenTelemetry and hosted tracing tools."
framework: llamaindex
language: python
---

# LlamaIndex Observability & Monitoring (Python)

> Verified against `llama-index-core` %%version=llama-index%% (checked %%versions-checked%%). Upstream: [releases](https://github.com/run-llama/llama_index/releases) · [PyPI](https://pypi.org/project/llama-index/)

LlamaIndex has two hook systems:

- **Instrumentation** (`llama_index.core.instrumentation`) — the current one. A dispatcher emits *spans* (nested timings around every query, retrieval, LLM call, tool call and workflow step) and *events* (LLM start/end, embedding start/end, retrieval results, agent tool calls). Integrations for OpenTelemetry and most tracing products build on it.
- **Callbacks** (`CallbackManager`) — the older system, still present and still the simplest way to count tokens.

---

## 1. OpenTelemetry

```bash
pip install llama-index-observability-otel opentelemetry-exporter-otlp
```

```python
from llama_index.observability.otel import LlamaIndexOpenTelemetry
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

instrumentor = LlamaIndexOpenTelemetry(
    service_name_or_resource="rag-api",
    span_exporter=OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces"),
    span_processor="batch",            # "simple" for local debugging
)
instrumentor.start_registering()       # call once at start-up, before building indexes/agents
```

Every query engine call, retrieval, LLM call and agent step now appears as a span in Jaeger, Tempo, Honeycomb, Datadog or any OTLP back end. Pass `tracer_provider=` to share an existing provider with the rest of your app (for example FastAPI's instrumentation).

## 2. Hosted tracing tools

Most tracing products ship a LlamaIndex integration; the one-line form uses `set_global_handler`:

```python
# pip install llama-index-callbacks-arize-phoenix
from llama_index.core import set_global_handler

set_global_handler("arize_phoenix")   # sends traces to a local or hosted Phoenix
```

Langfuse, Arize, Weights & Biases, OpenLLMetry and others document their own set-up (usually an OpenInference/OpenTelemetry instrumentor). Prefer the OpenTelemetry route when you already run a collector.

## 3. Custom event handlers

```python
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent
from llama_index.core.instrumentation.events.retrieval import RetrievalEndEvent

class Metrics(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        return "Metrics"

    def handle(self, event: BaseEvent, **kwargs) -> None:
        if isinstance(event, RetrievalEndEvent):
            print("retrieved", len(event.nodes), "nodes")
        elif isinstance(event, LLMChatEndEvent):
            print("llm call finished", event.span_id)

get_dispatcher().add_event_handler(Metrics())
```

Handlers run synchronously on the calling task — keep them cheap and push heavy work onto a queue.

## 4. Token counting

```python
import tiktoken
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4o").encode   # pick the tokenizer that matches your model family
)
Settings.callback_manager = CallbackManager([token_counter])

# ... run queries ...
print("embedding tokens:", token_counter.total_embedding_token_count)
print("LLM prompt tokens:", token_counter.prompt_llm_token_count)
print("LLM completion tokens:", token_counter.completion_llm_token_count)
token_counter.reset_counts()
```

The counts are estimates from the tokenizer; for billing-grade numbers read the provider's usage block from `LLMChatEndEvent.response.raw`.

## 5. Metrics worth exporting

| Metric | Type | Labels |
|---|---|---|
| Request latency | histogram | endpoint, index |
| Retrieval latency and node count | histogram | retriever type |
| LLM latency | histogram | model |
| Prompt / completion tokens | counter | model |
| Tool calls and tool errors | counter | tool name |
| Agent iterations per request | histogram | agent name |
| Cache hit rate (ingestion, semantic cache) | gauge | cache |

The [production guide](../llamaindex_production_guide/#monitoring-and-observability) shows these wired into Prometheus.

## 6. Logs

- Emit structured JSON logs with a request ID; put the same ID in the OTel span attributes so logs and traces join up.
- Log source node IDs (not their full text) for each answer — enough to reproduce, without copying documents into your log store.
- Redact prompts and tool arguments that may contain personal data before logging.

## 7. Debugging locally

```python
import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)   # very verbose: shows HTTP calls and prompts
```

For agents, iterate `handler.stream_events()` and print `ToolCall` / `ToolCallResult` events — see [streaming](../llamaindex_comprehensive_guide/#11-streaming-responses).

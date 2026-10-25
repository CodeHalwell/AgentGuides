---
title: "LlamaIndex Observability & Monitoring (Python)"
description: "Tracing, token accounting, metrics and logging for LlamaIndex using the instrumentation module, OpenTelemetry and hosted tracing tools."
framework: "llamaindex"
language: "python"
sidebar:
  label: "Observability"
---

> Verified against `llama-index-core` %%version=llama-index%% and `llama-index-observability-otel` (checked %%versions-checked%%). Upstream: [releases](https://github.com/run-llama/llama_index/releases) · [PyPI](https://pypi.org/project/llama-index/)

LlamaIndex has two hook systems:

- **Instrumentation** (`llama_index.core.instrumentation`) is the current one. A *dispatcher* emits **spans** (nested timings around every query, retrieval, embedding call, LLM call, tool call and workflow step) and **events** (point-in-time records such as "retrieval finished with these nodes" or "LLM call finished with this response"). The OpenTelemetry integration and most tracing products build on it.
- **Callbacks** (`CallbackManager`) are the older system. They are still present, and `TokenCountingHandler` and `set_global_handler("simple")` use them.

Both are global switches. Register handlers once at start-up, before you build indexes and agents, and they apply to everything in the process.

---

## 1. What gets instrumented

Spans are named `ClassName.method`. This is what one query against a `VectorStoreIndex` produces with a chat model such as `OpenAI` (the class name in the last lines is your LLM's):

```text
RetrieverQueryEngine.query
  RetrieverQueryEngine._query
    VectorIndexRetriever.retrieve
      VectorIndexRetriever._retrieve
        OpenAIEmbedding.get_query_embedding
          OpenAIEmbedding._get_query_embedding
    CompactAndRefine.synthesize
      CompactAndRefine.get_response
        TokenTextSplitter.split_text
        CompactAndRefine.get_response
          TokenTextSplitter.split_text
          DefaultRefineProgram.__call__
            OpenAI.predict
              OpenAI.chat
```

and one `FunctionAgent` run that calls a tool once (the second round, where the model answers, repeats the `setup_agent` to `parse_agent_output` group):

```text
FunctionAgent.run
  BaseWorkflowAgent.init_run
  BaseWorkflowAgent.setup_agent
  BaseWorkflowAgent.run_agent_step
    OpenAI._prepare_chat_with_tools
    OpenAI.astream_chat
  BaseWorkflowAgent.parse_agent_output
  BaseWorkflowAgent.call_tool
    FunctionTool.acall
  BaseWorkflowAgent.aggregate_tool_results
  ...
```

Events are attached to the span that was open when they fired. The ones you will use most, all importable from `llama_index.core.instrumentation.events.*`:

| Module | Events |
|---|---|
| `query` | `QueryStartEvent`, `QueryEndEvent` |
| `retrieval` | `RetrievalStartEvent`, `RetrievalEndEvent` (has `.nodes`) |
| `embedding` | `EmbeddingStartEvent`, `EmbeddingEndEvent` |
| `synthesis` | `SynthesizeStartEvent`, `SynthesizeEndEvent`, `GetResponseStartEvent`, `GetResponseEndEvent` |
| `rerank` | `ReRankStartEvent`, `ReRankEndEvent` |
| `llm` | `LLMChatStartEvent` / `LLMChatInProgressEvent` / `LLMChatEndEvent`, the same three for `LLMCompletion...`, and `LLMPredict...`, `LLMStructuredPredict...` |
| `exception` | `ExceptionEvent` |

Agents built on workflows also emit `WorkflowStepOutputEvent` and `WorkflowRunOutputEvent` (from `workflows.runtime.types.step_function`) for each step and for the run. `AgentToolCallEvent` still exists in `llama_index.core.instrumentation.events.agent`, but the workflow-based agents do not emit it; to see tool calls, look at the `BaseWorkflowAgent.call_tool` and `FunctionTool.acall` spans, or at the `ToolCall` and `ToolCallResult` events in the agent's own [event stream](/llamaindex-guide/python/llamaindex_comprehensive_guide/#11-streaming-responses).

---

## 2. OpenTelemetry

```bash
pip install llama-index-observability-otel opentelemetry-exporter-otlp
```

```python
from llama_index.observability.otel import LlamaIndexOpenTelemetry
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

instrumentor = LlamaIndexOpenTelemetry(
    service_name_or_resource="rag-api",
    span_exporter=OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces"),
    span_processor="batch",            # "simple" exports each span as it ends: use it for local debugging
)
instrumentor.start_registering()       # call once at start-up, before building indexes/agents
```

Every query engine call, retrieval, LLM call and agent step now appears as a span in Jaeger, Tempo, Honeycomb, Datadog or any other OTLP back end. The constructor takes:

| Argument | Default | Meaning |
|---|---|---|
| `span_exporter` | `ConsoleSpanExporter()` | Where spans go. Any OpenTelemetry `SpanExporter`. |
| `span_processor` | `"batch"` | `"batch"` or `"simple"`. |
| `service_name_or_resource` | a resource named `llamaindex.opentelemetry` | A service name string, or a full OpenTelemetry `Resource`. |
| `tracer_provider` | `None` | Reuse a provider you have already configured, so LlamaIndex spans share a trace with, for example, your FastAPI instrumentation. When `None`, a new provider is created. |
| `extra_span_processors` | `[]` | Additional processors added to the provider. |
| `debug` | `False` | Print a line whenever a span starts or ends or an event is recorded. |

`start_registering()` also installs the provider as the global OpenTelemetry tracer provider, so call it once per process: a second call logs "Overriding of current TracerProvider is not allowed" and keeps exporting to the first exporter. LlamaIndex events (LLM start and end, retrieval results and so on) are attached to their span as span events with the event's fields flattened into attributes. That includes query strings, retrieved text and model output, so treat the trace store as holding user data, or strip attributes in the collector.

To see what is emitted without running a collector, use the in-memory exporter from the OpenTelemetry SDK and print the span tree after a run:

```python
import asyncio

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.observability.otel import LlamaIndexOpenTelemetry

exporter = InMemorySpanExporter()
LlamaIndexOpenTelemetry(
    service_name_or_resource="demo", span_exporter=exporter, span_processor="simple"
).start_registering()


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


agent = FunctionAgent(tools=[add], llm=OpenAI(model="%%model=openai-small%%"))


async def main():
    await agent.run(user_msg="What is 2 + 3?")

    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}

    def depth(span) -> int:
        d = 0
        while span.parent is not None and span.parent.span_id in by_id:
            span, d = by_id[span.parent.span_id], d + 1
        return d

    for span in sorted(spans, key=lambda s: s.start_time):
        duration_ms = (span.end_time - span.start_time) / 1e6
        print(f"{'  ' * depth(span)}{span.name}  {duration_ms:.0f} ms")


asyncio.run(main())
```

---

## 3. Hosted tracing tools

Most tracing products ship a LlamaIndex integration. For some of them the one-line form is `set_global_handler`:

```python
# pip install llama-index-callbacks-arize-phoenix
from llama_index.core import set_global_handler

set_global_handler("arize_phoenix")   # sends traces to a Phoenix server (default endpoint http://127.0.0.1:6006/v1/traces)
```

`set_global_handler` accepts `"arize_phoenix"`, `"openinference"`, `"wandb"`, `"honeyhive"`, `"promptlayer"`, `"deepeval"`, `"argilla"`, `"langfuse"`, `"agentops"`, `"literalai"`, `"opik"` and `"simple"`. Each one except `"simple"` needs its own `llama-index-callbacks-*` package (the error message names it) and keyword arguments are passed to that package. The Phoenix handler exports with a synchronous span processor, so start Phoenix first or pass `endpoint=...`; otherwise each span waits on a failing connection. Where a vendor documents an OpenInference or OpenTelemetry instrumentor, prefer that, and prefer the OpenTelemetry route in section 2 when you already run a collector. Follow the vendor's documentation for set-up details, which change faster than this page.

---

## 4. Custom event handlers

An event handler sees every event in the process. This one counts retrievals and records token usage from each LLM call:

```python
from llama_index.core.callbacks.token_counting import get_tokens_from_response
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent, LLMCompletionEndEvent
from llama_index.core.instrumentation.events.retrieval import RetrievalEndEvent


class Metrics(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        return "Metrics"

    def handle(self, event: BaseEvent, **kwargs) -> None:
        if isinstance(event, RetrievalEndEvent):
            print("retrieved", len(event.nodes), "nodes")
        elif isinstance(event, (LLMChatEndEvent, LLMCompletionEndEvent)) and event.response is not None:
            prompt_tokens, completion_tokens = get_tokens_from_response(event.response)
            print("llm call", event.span_id, prompt_tokens, completion_tokens)


get_dispatcher().add_event_handler(Metrics())
```

Chat models (OpenAI, Anthropic and the agents) fire the `LLMChat*` events; models used through `complete()` fire `LLMCompletion*`. Match on both, as above, or you will see nothing from one of the two paths. `get_tokens_from_response` reads the provider's usage block (`raw["usage"]` or `usage_metadata`, falling back to `additional_kwargs`) and returns `(0, 0)` when the provider reported none. Streamed OpenAI responses only include usage if the request asked for it; if you see zeros on streamed calls, pass `additional_kwargs={"stream_options": {"include_usage": True}}` to the `OpenAI` constructor.

Handlers run synchronously on the calling task, so keep them cheap and push heavy work onto a queue. An exception raised inside `handle` is swallowed without a log line and does not fail the request, so wrap the body in `try`/`except` and log it yourself if you need to know about failures.

---

## 5. Token counting

`TokenCountingHandler` is a callback handler that totals tokens for LLM and embedding calls:

```python
import tiktoken
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.get_encoding("o200k_base").encode,   # used wherever the provider reports no usage
    token_budget=2_000_000,                                  # optional hard cap, see below
)
Settings.callback_manager = CallbackManager([token_counter])

# ... run queries ...
print("embedding tokens:", token_counter.total_embedding_token_count)
print("LLM prompt tokens:", token_counter.prompt_llm_token_count)
print("LLM completion tokens:", token_counter.completion_llm_token_count)
print("LLM total tokens:", token_counter.total_llm_token_count)
token_counter.reset_counts()
```

For LLM calls the handler uses the usage numbers the provider returned when there are any, and estimates with the tokenizer otherwise. Embedding counts are always tokenizer estimates, so they can differ slightly from the provider's bill. If you leave out `tokenizer`, it uses the `gpt-3.5-turbo` encoding from `tiktoken`, which suits OpenAI models of that generation and is only an approximation for others.

`token_budget` makes the handler raise `ValueError("Token budget exceeded! ...")` once total LLM tokens pass the limit, which stops a runaway loop. It counts the lifetime of the handler, applies to LLM tokens only, and the call that crosses the limit has already been paid for. Treat it as a circuit breaker, not an accounting tool.

---

## 6. Metrics worth exporting

| Metric | Type | Labels |
|---|---|---|
| Request latency | histogram | endpoint, index |
| Retrieval latency and node count | histogram | retriever type |
| LLM latency | histogram | model |
| Prompt / completion tokens | counter | model |
| Tool calls and tool errors | counter | tool name |
| Agent iterations per request | histogram | agent name |
| Cache hit rate (ingestion, semantic cache) | gauge | cache |

The [production guide](/llamaindex-guide/python/llamaindex_production_guide/#5-observability-and-cost) shows OpenTelemetry and token-based cost tracking in a deployed service.

## 7. Logs

- Emit structured JSON logs with a request ID; put the same ID in the OpenTelemetry span attributes so logs and traces join up.
- Log source node IDs (not their full text) for each answer: enough to reproduce, without copying documents into your log store.
- Redact prompts and tool arguments that may contain personal data before logging.

## 8. Debugging locally

The quickest look at what a query sends to the LLM is the `"simple"` global handler, which prints each prompt and completion:

```python
from llama_index.core import set_global_handler

set_global_handler("simple")
```

For a timing breakdown per stage, `LlamaDebugHandler` prints a trace tree at the end of each query and can be queried afterwards:

```python
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

debug = LlamaDebugHandler(print_trace_on_end=True)
Settings.callback_manager = CallbackManager([debug])

# ... run a query ...
print(debug.get_event_time_info("llm"))     # total and average seconds, and call count, for LLM events
```

The trace lists `QUERY`, `RETRIEVE`, `EMBEDDING`, `SYNTHESIZE`, `TEMPLATING` and `LLM` with durations. For everything the HTTP clients send, raise the log level:

```python
import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)   # very verbose: shows HTTP calls and prompts
```

For agents, iterate `handler.stream_events()` and print `ToolCall` / `ToolCallResult` events; see [streaming](/llamaindex-guide/python/llamaindex_comprehensive_guide/#11-streaming-responses).

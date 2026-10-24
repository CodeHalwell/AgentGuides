---
title: "Strands Agents (Python): Comprehensive Guide"
description: "The Python Strands Agents SDK from the ground up: the agent loop, Agent options, model providers, tools, structured output, streaming, conversation management, state, sessions, limits and errors."
framework: "strands"
language: "python"
sidebar:
  label: "Comprehensive guide"
  order: 1
---

Checked against `strands-agents` %%version=strands-agents%% and `strands-agents-tools` %%version=strands-agents-tools%% (%%versions-checked%%). Every example on this page was run against the installed package with the Bedrock client replaced by an offline stub, so the code paths are real but the model replies are canned. See [Testing without the cloud](/strands-agents-guide/python/strands_production_guide/#testing-without-the-cloud) for the technique.

Topic pages go deeper on [tools and MCP](/strands-agents-guide/python/strands_tools_and_mcp/), [hooks, interventions and interrupts](/strands-agents-guide/python/strands_hooks_and_interrupts/), [multi-agent systems and A2A](/strands-agents-guide/python/strands_multi_agent/) and [observability and evals](/strands-agents-guide/python/strands_observability_and_evals/).

## Installation

```bash
pip install "strands-agents==%%version=strands-agents%%" "strands-agents-tools==%%version=strands-agents-tools%%"
```

Python 3.10 or later. Model providers other than Bedrock, and some features, are optional extras:

| Extra | Adds |
|-------|------|
| `anthropic` | `AnthropicModel` (Anthropic API) |
| `openai` | `OpenAIModel`, `OpenAIResponsesModel` |
| `gemini` | `GeminiModel` |
| `litellm` | `LiteLLMModel` |
| `ollama` | `OllamaModel` |
| `mistral`, `llamaapi`, `writer`, `sagemaker` | `MistralModel`, `LlamaAPIModel`, `WriterModel`, `SageMakerAIModel` |
| `a2a` | `A2AServer`, `A2AAgent` |
| `otel` | OTLP trace exporter |
| `bidi`, `bidi-openai`, `bidi-google` | `BidiAgent` for real-time voice |
| `web-fetch` | the vended `web_fetch` tool |

```bash
pip install "strands-agents[anthropic,openai,otel]==%%version=strands-agents%%"
```

`LlamaCppModel` needs no extra (it talks HTTP to a llama.cpp server).

### AWS credentials

Bedrock is the default provider. Strands uses the standard boto3 credential chain (environment variables, `~/.aws` profiles, SSO, instance or task roles). Set a Region, and enable access to the models you plan to use in the Bedrock console:

```bash
aws configure          # or: export AWS_PROFILE=...
export AWS_REGION=us-west-2
```

A first agent:

```python
from strands import Agent, tool


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text.

    Args:
        text: The text to count.
    """
    return len(text.split())


agent = Agent(
    model="us.anthropic.%%model=anthropic%%",  # Bedrock cross-Region inference profile
    system_prompt="You are a concise assistant.",
    tools=[word_count],
)

result = agent("How many words are in 'the quick brown fox'?")
print(result.stop_reason)   # end_turn
print(str(result))          # the final text
```

By default an agent prints the streamed response to stdout (`PrintingCallbackHandler`). Pass `callback_handler=None` to silence it in servers and tests.

## The agent loop

Strands is model-driven: you give the agent a model, a system prompt and tools, and the model decides what to do. One call to the agent runs this loop:

1. Append the user input to `agent.messages`.
2. Call the model with the history, the system prompt and the tool specs.
3. If the model asked for tools (`stop_reason == "tool_use"`), run them (concurrently by default), append the results as a user message, and go back to step 2.
4. Otherwise stop and return an `AgentResult`.

Each pass through steps 2–3 is an *event loop cycle*. There is no fixed iteration cap; bound a run with `limits=` (below) or with hooks. A [diagram of the loop](/strands-agents-guide/strands_diagrams/#the-agent-loop) shows where hooks fire.

What the loop does around the edges:

- **Tool errors** do not end the run. An exception in a tool becomes a tool result with `status: "error"`, and the model sees the message and can recover.
- **Throttling** is retried with exponential backoff (`ModelRetryStrategy`, six attempts by default).
- **Context overflow** (`ContextWindowOverflowException` from the provider) makes the conversation manager trim the history and the call is retried.
- **`max_tokens`** raises `MaxTokensReachedException`; the partial message stays in the history, so you can raise `max_tokens` or call the agent again.

### AgentResult

```python
from strands import Agent

agent = Agent(model="us.anthropic.%%model=anthropic%%", callback_handler=None)
result = agent("Name three Yorkshire towns.")

print(result.stop_reason)                         # "end_turn"
print(result.message["content"][0]["text"])       # the final assistant message
print(result.metrics.accumulated_usage)           # {'inputTokens': ..., 'outputTokens': ..., 'totalTokens': ...}
print(result.metrics.cycle_count)                 # number of loop cycles
print(len(agent.messages))                        # history kept on the agent
```

| Field | Meaning |
|-------|---------|
| `stop_reason` | `end_turn`, `max_tokens`, `stop_sequence`, `guardrail_intervened`, `content_filtered`, `interrupt`, `cancelled`, `limit_turns`, `limit_total_tokens`, `limit_output_tokens`, `checkpoint`, `tool_use` |
| `message` | The last assistant message |
| `metrics` | `EventLoopMetrics`: usage, cycle count, per-tool metrics, traces |
| `structured_output` | The validated Pydantic instance when you asked for one |
| `interrupts` | Pending interrupts when `stop_reason == "interrupt"` |
| `state` | Final request state |

`str(result)` returns the text of the final message (with a trailing newline), so `str(agent(...)).strip()` is the usual way to get a plain answer.

## The Agent class

The constructor takes keyword arguments. The ones you will use most:

| Argument | Purpose |
|----------|---------|
| `model` | A `Model` instance, a Bedrock model ID string, or a `ModelRouter`. Defaults to `BedrockModel()` |
| `system_prompt` | A string, or a list of system content blocks (for cache points) |
| `tools` | `@tool` functions, tool modules, `AgentTool` objects, `MCPClient`s and other tool providers, or other `Agent`s |
| `name`, `description`, `agent_id` | Identity; used when the agent becomes a tool, a swarm member or an A2A server, and as the session key |
| `messages` | Initial history |
| `state` | JSON-serialisable application state (not sent to the model) |
| `callback_handler` | Receives streaming events; `None` silences output |
| `conversation_manager` | How history is trimmed (default `SlidingWindowConversationManager`) |
| `session_manager` | Persists messages and state |
| `structured_output_model` | Default Pydantic output type for every call |
| `hooks`, `plugins`, `interventions` | Lifecycle callbacks, bundled extensions, and policy handlers |
| `retry_strategy` | `ModelRetryStrategy(...)`, or `None` to disable retries |
| `tool_executor` | `ConcurrentToolExecutor()` (default) or `SequentialToolExecutor()` |
| `trace_attributes` | Extra OpenTelemetry span attributes |
| `aux_model` | A cheaper model for side calls (summarisation, judges, classifiers) |

`agent.tool_names` lists the registered tools, `agent.messages` is the history (a list of Bedrock-style message dicts), and `agent.state` is the state store.

### Ways to call an agent

```python
import asyncio

from strands import Agent

agent = Agent(model="us.anthropic.%%model=anthropic%%", callback_handler=None)

# Synchronous: blocks until the loop finishes
result = agent("Summarise the plot of Hamlet in one sentence.")

# Asynchronous
async def main() -> None:
    result = await agent.invoke_async("And Macbeth?")
    print(result)

    # Streaming: an async iterator of events, ending with {"result": AgentResult}
    async for event in agent.stream_async("And King Lear?"):
        if "data" in event:
            print(event["data"], end="", flush=True)
        elif "result" in event:
            print("\nstop:", event["result"].stop_reason)

asyncio.run(main())
```

The prompt can be a string, a list of content blocks (text, images, documents), a list of full messages, or `None` to continue from the existing history. A multimodal prompt:

```python
from pathlib import Path

from strands import Agent

Path("chart.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)  # stand-in image

agent = Agent(model="us.anthropic.%%model=anthropic%%", callback_handler=None)
result = agent([
    {"text": "What does this chart show?"},
    {"image": {"format": "png", "source": {"bytes": Path("chart.png").read_bytes()}}},
])
```

Documents use `{"document": {"format": "pdf", "name": "report", "source": {"bytes": ...}}}`.

An `Agent` handles one invocation at a time. A second concurrent call raises `ConcurrencyException`; create one agent per conversation (they are cheap) rather than sharing one across requests.

## Model providers

### Amazon Bedrock

A string `model` becomes `BedrockModel(model_id=...)`. Use `BedrockModel` directly for settings:

```python
from strands import Agent
from strands.models import BedrockModel, CacheConfig

model = BedrockModel(
    model_id="us.anthropic.%%model=anthropic%%",
    region_name="eu-west-1",                    # or boto_session=boto3.Session(...)
    max_tokens=4096,
    temperature=0.2,
    cache_config=CacheConfig(strategy="auto"),  # prompt caching for system prompt and tools
    context_window_limit=200_000,               # used by proactive context management
)

agent = Agent(model=model, callback_handler=None)
model.update_config(temperature=0.0)            # change settings between calls
```

**Model IDs.** Claude models on Bedrock have IDs such as `anthropic.%%model=anthropic%%`. Most are called through a cross-Region inference profile, which adds a geography prefix: `us.`, `eu.`, `apac.` or `global.` (for example `us.anthropic.%%model=anthropic-small%%`). Which prefixes exist depends on the model and Region; check the [Bedrock model cards](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html).

If you omit `model`, Strands uses its built-in default (`global.anthropic.claude-sonnet-4-6` in this release) and logs a warning that the default can change. Set the model explicitly in anything you deploy.

Other `BedrockModel` options worth knowing:

| Option | Use |
|--------|-----|
| `guardrail_id`, `guardrail_version`, `guardrail_trace`, `guardrail_redact_input`, `guardrail_redact_output` | Apply a Bedrock Guardrail to every call (see [below](#bedrock-guardrails)) |
| `additional_request_fields` | Model-specific fields, for example Claude extended thinking: `{"thinking": {"type": "enabled", "budget_tokens": 4000}}` |
| `stop_sequences`, `top_p` | Sampling controls |
| `streaming=False` | Use `Converse` instead of `ConverseStream` (for models without streaming tool use) |
| `boto_client_config` | A `botocore.config.Config` (timeouts, retries, connection pool) |
| `api_key` | A Bedrock API key instead of SigV4 credentials |
| `service_tier` | Bedrock service tier |

The deprecated `cache_prompt` and `cache_tools` options still work but `cache_config=CacheConfig(...)` replaces them.

#### Bedrock Guardrails

```python
from strands import Agent
from strands.models import BedrockModel

model = BedrockModel(
    model_id="us.anthropic.%%model=anthropic%%",
    guardrail_id="gr-abc123",
    guardrail_version="1",
    guardrail_trace="enabled",
    guardrail_redact_output=True,          # replace blocked output in the history
)
agent = Agent(model=model, callback_handler=None)
```

When the guardrail blocks, `result.stop_reason` is `guardrail_intervened` and, by default, the offending user input in `agent.messages` is replaced with a redaction message.

### Anthropic, OpenAI and others

Each provider is a `Model` class with its own client arguments and a `params` (or flat) config:

```python
import os

from strands import Agent
from strands.models.anthropic import AnthropicModel
from strands.models.litellm import LiteLLMModel
from strands.models.ollama import OllamaModel
from strands.models.openai import OpenAIModel

claude = AnthropicModel(
    client_args={"api_key": os.environ.get("ANTHROPIC_API_KEY", "sk-ant-...")},
    model_id="%%model=anthropic%%",
    max_tokens=4096,
    params={"temperature": 0.2},
)

gpt = OpenAIModel(
    client_args={"api_key": os.environ.get("OPENAI_API_KEY", "sk-...")},
    model_id="%%model=openai%%",
    params={"max_completion_tokens": 2000},
)

# Any OpenAI-compatible server (vLLM, LM Studio, a gateway) via base_url
local_compatible = OpenAIModel(
    client_args={"api_key": "unused", "base_url": "http://localhost:8000/v1"},
    model_id="my-model",
)

anything = LiteLLMModel(model_id="mistral/%%model=mistral%%", params={"max_tokens": 1000})

llama = OllamaModel(host="http://localhost:11434", model_id="llama3.2", temperature=0.3)

agent = Agent(model=claude, callback_handler=None)
```

| Class | Module | Constructor notes |
|-------|--------|-------------------|
| `BedrockModel` | `strands.models` | `model_id`, `region_name` or `boto_session`, flat config |
| `AnthropicModel` | `strands.models.anthropic` | `client_args` (to `anthropic.AsyncAnthropic`), `model_id`, `max_tokens` (required), `params` |
| `OpenAIModel` | `strands.models.openai` | Chat Completions. `client_args` or `client=`, `model_id`, `params` |
| `OpenAIResponsesModel` | `strands.models.openai_responses` | Responses API; `stateful=True` keeps conversation state server-side |
| `GeminiModel` | `strands.models.gemini` | `client_args` (to `google.genai.Client`), `model_id`, `params` |
| `LiteLLMModel` | `strands.models.litellm` | `model_id` in LiteLLM's `provider/model` form, `params` |
| `OllamaModel` | `strands.models.ollama` | `host` (positional), `model_id`, `temperature`, `max_tokens`, `options` |
| `MistralModel` | `strands.models.mistral` | `api_key` (or `MISTRAL_API_KEY`), `model_id`, `max_tokens` |
| `LlamaCppModel` | `strands.models.llamacpp` | `base_url` of a llama.cpp server |
| `LlamaAPIModel`, `WriterModel`, `SageMakerAIModel` | `strands.models.llamaapi`, `.writer`, `.sagemaker` | Provider-specific |

The Anthropic and OpenAI model IDs are the provider's own (`%%model=anthropic%%`, `%%model=openai%%`), not Bedrock IDs.

### Routing and fallback

`ModelRouter` wraps several models. The default `FallbackStrategy` uses the first, and moves to the next when a call fails (after that model's retries):

```python
from strands import Agent
from strands.models import BedrockModel, ModelRouter
from strands.models.anthropic import AnthropicModel

router = ModelRouter([
    BedrockModel(model_id="us.anthropic.%%model=anthropic%%"),
    AnthropicModel(client_args={"api_key": "sk-ant-..."}, model_id="%%model=anthropic%%", max_tokens=4096),
])
agent = Agent(model=router, callback_handler=None)
```

`ClassifierStrategy` picks a model per call instead (for example a small model for easy requests). `aux_model=` on the agent is the simpler tool for cost: it moves Strands' own side calls (context summarisation, memory extraction, judges) to a cheaper model while the main loop keeps `model`.

### Writing a provider

A provider subclasses `strands.models.Model` and implements `stream()` (an async generator of Bedrock-style stream events), `update_config()`, `get_config()` and `structured_output()`. The scripted test model on the [production page](/strands-agents-guide/python/strands_production_guide/#a-scripted-model) is a complete, minimal example; wrapping a new HTTP API follows the same shape, translating its chunks into `messageStart`, `contentBlockStart`/`Delta`/`Stop`, `messageStop` and `metadata` events.

## Tools

Any function decorated with `@tool` is a tool. Its name is the function name, its description is the docstring, and the input schema comes from the type hints and the `Args:` section:

```python
from typing import Literal

from strands import Agent, tool


@tool
def convert_temperature(value: float, to: Literal["C", "F"]) -> str:
    """Convert a temperature between Celsius and Fahrenheit.

    Args:
        value: The temperature to convert.
        to: The unit to convert to.
    """
    result = value * 9 / 5 + 32 if to == "F" else (value - 32) * 5 / 9
    return f"{result:.1f}°{to}"


agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[convert_temperature], callback_handler=None)
print(agent.tool_names)                              # ['convert_temperature']
print(convert_temperature(21, "F"))                  # still a normal function: 69.8°F
print(agent.tool.convert_temperature(value=0, to="F"))  # direct call, bypassing the model
```

Tools can be async, can stream progress by yielding, can read the calling agent and per-request state through `ToolContext`, and can come from MCP servers or `strands-agents-tools`. All of that is on the [tools and MCP page](/strands-agents-guide/python/strands_tools_and_mcp/).

## Structured output

Pass a Pydantic model as `structured_output_model`, on the agent (a default for every call) or per call. The validated instance is on `result.structured_output`:

```python
from pydantic import BaseModel, Field
from strands import Agent


class Invoice(BaseModel):
    """Fields extracted from an invoice."""

    invoice_id: str
    total: float = Field(description="Total including VAT")
    currency: str = Field(description="ISO 4217 code, for example GBP")


agent = Agent(model="us.anthropic.%%model=anthropic%%", callback_handler=None)
result = agent("Extract: invoice INV-77, total £1,200.00 including VAT", structured_output_model=Invoice)
invoice = result.structured_output          # an Invoice
print(invoice.invoice_id, invoice.total, invoice.currency)
```

How it works: Strands adds a tool whose input schema is your model. The model can call other tools first; when it finishes without calling the output tool, Strands asks it once more, forcing that tool (`structured_output_prompt=` customises the request). Some providers enforce the schema natively instead. If no valid output arrives, `StructuredOutputException` is raised:

```python
from pydantic import BaseModel
from strands import Agent
from strands.types.exceptions import StructuredOutputException


class Sentiment(BaseModel):
    label: str
    score: float


agent = Agent(model="us.anthropic.%%model=anthropic%%", callback_handler=None)
try:
    sentiment = agent("I love this product", structured_output_model=Sentiment).structured_output
except StructuredOutputException as exc:
    print("no structured output:", exc)
```

`Agent.structured_output(Model, prompt)` from earlier releases still exists but is deprecated; use `structured_output_model=`.

## Streaming

There are two ways to receive events while the loop runs.

**`stream_async`** yields dictionaries. The useful keys:

| Key | When |
|-----|------|
| `data` | A text delta from the model (`delta` holds the raw delta) |
| `reasoningText` | A reasoning (extended thinking) delta |
| `current_tool_use` | The model is streaming a tool call (`name`, `toolUseId`, partial `input`) |
| `tool_stream_event` | A streaming tool yielded progress |
| `message` | A complete message was added (assistant turn or tool results) |
| `event` | The raw provider stream event |
| `result` | The final `AgentResult` (last event) |
| `init_event_loop`, `start`, `start_event_loop` | Lifecycle markers |

```python
import asyncio

from strands import Agent, tool


@tool
def lookup_stock(sku: str) -> int:
    """Return units in stock.

    Args:
        sku: Product SKU.
    """
    return 12


agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[lookup_stock], callback_handler=None)


async def main() -> None:
    async for event in agent.stream_async("How many A-1 widgets are in stock?"):
        if "data" in event:
            print(event["data"], end="", flush=True)
        elif "current_tool_use" in event and event["current_tool_use"].get("name"):
            pass  # a tool call is being streamed
        elif "message" in event and event["message"]["role"] == "user":
            print("\n[tool results returned]")
        elif "result" in event:
            print("\n[done]", event["result"].stop_reason)


asyncio.run(main())
```

**Callback handlers** receive the same events as keyword arguments, for code that calls the agent synchronously:

```python
from strands import Agent


def on_event(**kwargs) -> None:
    if "data" in kwargs:
        print(kwargs["data"], end="")
    if "current_tool_use" in kwargs and kwargs["current_tool_use"].get("name"):
        print(f"\n[calling {kwargs['current_tool_use']['name']}]")


agent = Agent(model="us.anthropic.%%model=anthropic%%", callback_handler=on_event)
agent("Tell me a short joke")
```

`strands.handlers.callback_handler` has `PrintingCallbackHandler` (the default), `null_callback_handler` and `CompositeCallbackHandler(*handlers)`.

For a web server, iterate `stream_async` and forward `data` events, as in the [streaming recipe](/strands-agents-guide/python/strands_recipes/#streaming-over-http-with-fastapi). Hooks are the better tool for side effects such as audit logs, because they fire whether or not anyone is consuming the stream.

## Conversation management

`agent.messages` grows with every turn. The conversation manager decides what to keep:

| Manager | Behaviour |
|---------|-----------|
| `SlidingWindowConversationManager(window_size=40)` | Default. After each cycle, drops the oldest messages beyond `window_size`, never splitting a tool call from its result. On a context overflow it first truncates oversized tool results (`should_truncate_results=True`), then trims |
| `SummarizingConversationManager(summary_ratio=0.3, preserve_recent_messages=10)` | Does nothing until the context overflows, then replaces the oldest messages with a model-written summary. `summarization_agent=` or `summarization_system_prompt=` customise it |
| `NullConversationManager()` | Never trims; overflow errors propagate |

```python
from strands import Agent
from strands.agent.conversation_manager import (
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)

chat = Agent(
    model="us.anthropic.%%model=anthropic%%",
    conversation_manager=SlidingWindowConversationManager(window_size=20, pin_first=1),
    callback_handler=None,
)

support = Agent(
    model="us.anthropic.%%model=anthropic%%",
    conversation_manager=SummarizingConversationManager(summary_ratio=0.4, preserve_recent_messages=8),
    aux_model="us.anthropic.%%model=anthropic-small%%",   # summaries come from the cheaper model
    callback_handler=None,
)
```

`pin_first=N` keeps the first N messages (for example a document the user pasted at the start). `proactive_compression=True` makes either manager compress before the model call once 70% of the model's `context_window_limit` is used (`{"compression_threshold": 0.8}` sets another ratio), instead of waiting for the provider to report an overflow.

The newer `context_manager=` argument replaces the conversation manager with a strategy pipeline: `"auto"` truncates large tool results and summarises at 85% utilisation, `"agentic"` gives the model tools to manage its own context, and `False` turns context management off. Set `context_window_limit` on the model when you use it, otherwise a 200,000-token default is assumed.

## State

There are two kinds of state, and neither is shown to the model.

**`agent.state`** is durable, JSON-serialisable key-value data that lives with the agent and is saved by the session manager:

```python
from strands import Agent

agent = Agent(model="us.anthropic.%%model=anthropic%%", state={"tier": "gold"}, callback_handler=None)
agent.state.set("cart", ["A-1", "B-2"])
print(agent.state.get("cart"), agent.state.get("tier"))
print(agent.state.get())            # the whole dict
agent.state.delete("cart")
```

Values must be JSON-serialisable; `set` raises `ValueError` otherwise.

**`invocation_state`** is per call: pass anything (including non-serialisable objects such as a database connection) and it reaches tools, hooks and sub-agents for that call only:

```python
from strands import Agent, ToolContext, tool


@tool(context=True)
def my_orders(tool_context: ToolContext) -> str:
    """List the current customer's orders."""
    customer = tool_context.invocation_state["customer_id"]
    tier = tool_context.agent.state.get("tier")
    return f"{customer} ({tier}): ORD-1, ORD-2"


agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[my_orders], state={"tier": "gold"}, callback_handler=None)
agent("What have I ordered?", invocation_state={"customer_id": "C-456"})
```

This is where user IDs, tenant IDs and credential handles belong: the model cannot see or change them.

## Sessions

A session manager saves `agent.messages`, `agent.state` and the conversation manager's state, and restores them when an agent with the same session ID is created, so a conversation survives process restarts.

```python
from strands import Agent
from strands.session import FileSessionManager

def make_agent(session_id: str) -> Agent:
    return Agent(
        model="us.anthropic.%%model=anthropic%%",
        session_manager=FileSessionManager(session_id=session_id, storage_dir="./sessions"),
        callback_handler=None,
    )

make_agent("user-42")("My name is Priya.")
later = make_agent("user-42")            # a new process could do this
print(len(later.messages))               # 2: the earlier turn was restored
```

| Manager | Storage |
|---------|---------|
| `FileSessionManager(session_id, storage_dir=None)` | One JSON file per message on local disk |
| `S3SessionManager(session_id, bucket, prefix="", region_name=None, boto_session=None)` | The same layout in S3 |
| `SnapshotSessionManager(session_id, storage=...)` | The whole agent as one versioned snapshot per save, through a `Storage` backend (`LocalFileStorage`, `S3Storage`, `InMemoryStorage` from `strands.storage`); supports immutable checkpoints with `snapshot_trigger=` |
| `AgentCoreMemorySessionManager` (in `bedrock-agentcore`) | Amazon Bedrock AgentCore Memory, with long-term memory retrieval |
| `RepositorySessionManager(session_id, session_repository)` | Your own `SessionRepository` (for example DynamoDB or Redis) |

```python
from strands import Agent
from strands.session import S3SessionManager, SnapshotSessionManager
from strands.storage import S3Storage

per_message = S3SessionManager(session_id="user-42", bucket="agent-sessions", prefix="support/")

snapshots = SnapshotSessionManager("user-42", storage=S3Storage("agent-sessions", prefix="snapshots/"))

agent = Agent(model="us.anthropic.%%model=anthropic%%", session_manager=snapshots, callback_handler=None)
agent("Hello")
```

The message-log managers write one object per message; the snapshot manager writes the whole agent atomically and is the one Strands recommends for new agents. Session IDs and agent IDs must not contain path separators. Members of a `Swarm` or `Graph` cannot have their own session manager; give one to the swarm or graph instead.

### AgentCore Memory

For managed short- and long-term memory on AWS, use the session manager in `bedrock-agentcore`:

```python
# pip install "bedrock-agentcore==%%version=bedrock-agentcore%%"
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from strands import Agent

config = AgentCoreMemoryConfig(
    memory_id="CustomerMemory-abc123",
    session_id="session-123",
    actor_id="customer-42",
    retrieval_config={"/facts/{actorId}/": RetrievalConfig(top_k=5, relevance_score=0.3)},
)

agent = Agent(
    model="us.anthropic.%%model=anthropic%%",
    session_manager=AgentCoreMemorySessionManager(config, region_name="us-west-2"),
)
```

Turns are written to AgentCore Memory as events; long-term records extracted by the memory's strategies are retrieved and added to the context on each turn. Creating the memory resource itself is covered in the [AgentCore guide](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/).

## Limits, cancellation and concurrency

```python
import threading

from strands import Agent

agent = Agent(model="us.anthropic.%%model=anthropic%%", callback_handler=None)

# Budget caps for one call; the loop stops cleanly at the next turn boundary
result = agent("Research this thoroughly", limits={"turns": 8, "total_tokens": 50_000})
if result.stop_reason.startswith("limit_"):
    print("stopped early:", result.stop_reason)

# Cancellation from outside (client disconnect, timeout)
stop = threading.Event()
timer = threading.Timer(30, stop.set)
timer.start()
try:
    result = agent("Write a long report", cancel_signal=stop)   # stop_reason == "cancelled" if it fired
finally:
    timer.cancel()
```

`agent.cancel()` cancels the current call from another thread or task. Token limits are soft: a single large response can overshoot by one turn. Limits are per call, not cumulative.

`idempotency_token=` deduplicates retried requests: a second call with the same token while the first is running waits for and returns the first call's result.

## Errors

Exceptions live in `strands.types.exceptions`:

| Exception | Raised when |
|-----------|-------------|
| `ModelThrottledException` | The provider is still throttling after all retries |
| `ContextWindowOverflowException` | The input does not fit and the conversation manager could not reduce it |
| `MaxTokensReachedException` | The model hit `max_tokens` mid-response |
| `StructuredOutputException` | No valid structured output was produced |
| `ConcurrencyException` | A second call was made while one is running |
| `MCPClientInitializationError` | An MCP server could not be started |
| `SessionException`, `SnapshotException` | Session storage failed |
| `EventLoopException` | Wraps other failures inside the loop |

```python
import logging

from strands import Agent
from strands.types.exceptions import (
    ContextWindowOverflowException,
    MaxTokensReachedException,
    ModelThrottledException,
)

log = logging.getLogger(__name__)
agent = Agent(model="us.anthropic.%%model=anthropic%%", callback_handler=None)


def answer(text: str) -> str:
    try:
        return str(agent(text)).strip()
    except ModelThrottledException:
        log.warning("still throttled after retries")
        raise
    except (ContextWindowOverflowException, MaxTokensReachedException) as exc:
        log.warning("context or output limit: %s", exc)
        raise
```

Strands already retries throttling, so do not wrap every call in a second retry loop; adjust `retry_strategy=ModelRetryStrategy(max_attempts=..., initial_delay=..., max_delay=...)` instead. Logging uses the `strands` logger: `logging.getLogger("strands").setLevel(logging.DEBUG)` shows requests and tool calls.

## Other features

These exist in this release and are worth knowing about, but are newer and change more often than the core:

- **Plugins and Agent Skills** (`plugins=[AgentSkills(skills=["./skills"])]`): see [hooks and interrupts](/strands-agents-guide/python/strands_hooks_and_interrupts/#plugins).
- **Sandboxes** (`sandbox=DockerSandbox(...)`): route the vended shell and file tools into a container. See [tools](/strands-agents-guide/python/strands_tools_and_mcp/#sandboxes).
- **Memory manager** (`memory_manager=MemoryManager(stores=[...])`): cross-session memory with extraction, backed by stores such as `BedrockKnowledgeBaseStore`.
- **Background tasks** (`background_tasks=True`): the model can start tools in the background and receive their results later.
- **Checkpointing** (`checkpointing=True`): the loop pauses at cycle boundaries with `stop_reason == "checkpoint"`, for durable execution engines.
- **Bidirectional streaming** (`strands.bidi.BidiAgent`): real-time voice and text over a persistent connection with Nova Sonic, Gemini Live or OpenAI Realtime models.

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | 1.62.0 | New page in the standalone Strands guide, replacing the Strands page in the Amazon Bedrock guide. Adds the agent loop, `AgentResult`, provider table, routing, streaming events, conversation and context managers, snapshot sessions, limits and cancellation. All examples run against an offline Bedrock stub. |

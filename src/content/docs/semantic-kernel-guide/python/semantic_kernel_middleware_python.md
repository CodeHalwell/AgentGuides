---
title: "Semantic Kernel Middleware (Python)"
description: "Guardrails, redaction, approval and telemetry in Semantic Kernel Python using kernel filters."
framework: semantic-kernel
language: python
---

# Semantic Kernel Middleware (Python)

> Verified against `semantic-kernel` %%version=semantic-kernel%% (checked %%versions-checked%%).

Semantic Kernel's middleware mechanism is **filters**: async functions that wrap a step and call `await next(context)` to continue. There are three kinds:

| Filter type | Wraps | Typical uses |
|---|---|---|
| `FilterTypes.PROMPT_RENDERING` | Rendering a prompt template, before it is sent | Redaction, prompt-injection checks, adding policy text |
| `FilterTypes.FUNCTION_INVOCATION` | Every kernel function call (prompt or native) | Logging, timing, caching, input validation, output redaction |
| `FilterTypes.AUTO_FUNCTION_INVOCATION` | Each function the model asks to call during function calling | Approval gates, blocking tools, stopping the tool loop |

Filters run for agents too, because agents invoke functions through their kernel.

## Input guardrail on rendered prompts

```python
import re
from semantic_kernel import Kernel
from semantic_kernel.filters import FilterTypes, PromptRenderContext

kernel = Kernel()
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

@kernel.filter(FilterTypes.PROMPT_RENDERING)
async def redact_prompt(context: PromptRenderContext, next):
    await next(context)                                   # render first
    context.rendered_prompt = SSN.sub("***-**-****", context.rendered_prompt or "")
```

## Output redaction and timing on function calls

```python
import time, logging
from semantic_kernel.filters import FunctionInvocationContext
from semantic_kernel.functions import FunctionResult

log = logging.getLogger("sk")

@kernel.filter(FilterTypes.FUNCTION_INVOCATION)
async def time_and_redact(context: FunctionInvocationContext, next):
    start = time.perf_counter()
    try:
        await next(context)
    finally:
        log.info("%s.%s took %.0f ms", context.function.plugin_name, context.function.name,
                 (time.perf_counter() - start) * 1000)
    if context.result is not None:
        # Native functions return their own value; prompt functions return chat message
        # content objects — str() gives the text in both cases.
        context.result = FunctionResult(
            function=context.function.metadata,
            value=SSN.sub("***-**-****", str(context.result)),
        )
```

To short-circuit — for example a cache hit — set `context.result` and return without calling `next`.

## Approval gate for tool calls

```python
from semantic_kernel.filters import AutoFunctionInvocationContext

SENSITIVE = {"delete_account", "send_payment"}

@kernel.filter(FilterTypes.AUTO_FUNCTION_INVOCATION)
async def require_approval(context: AutoFunctionInvocationContext, next):
    if context.function.name in SENSITIVE and not await approved_by_human(context):
        context.function_result = FunctionResult(
            function=context.function.metadata,
            value="The user declined this action. Do not retry it.",
        )
        return                                             # skip the call; the model sees the message
    await next(context)

async def approved_by_human(context: AutoFunctionInvocationContext) -> bool:
    print(f"Approve {context.function.name}({dict(context.arguments or {})})? [y/N]")
    return input().strip().lower() == "y"
```

Set `context.terminate = True` to stop the whole function-calling loop instead of just skipping one call.

## Ordering and registration

- Filters of the same type run in registration order, each wrapping the next (onion model).
- `kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, fn)` is the non-decorator form; `kernel.remove_filter(...)` removes one.
- Build a fresh kernel per request (or per tenant) if filters carry request-specific state.

## Telemetry

Semantic Kernel emits OpenTelemetry spans and metrics for function invocations and model calls. Configure a tracer provider and exporter as usual, then add attributes from a filter:

```python
from opentelemetry import trace

@kernel.filter(FilterTypes.FUNCTION_INVOCATION)
async def tag_span(context: FunctionInvocationContext, next):
    span = trace.get_current_span()
    span.set_attribute("app.tenant", (context.arguments or {}).get("tenant", "unknown"))
    await next(context)
```

Set `SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS=true` to emit GenAI semantic-convention spans for model calls (`..._SENSITIVE=true` also records prompts and completions — development only).

## Testing

- Filters are plain async functions: call them directly with a hand-built context and a fake `next`.
- For integration tests, register the filters on a kernel whose chat service is a stub and assert on what reaches the service.

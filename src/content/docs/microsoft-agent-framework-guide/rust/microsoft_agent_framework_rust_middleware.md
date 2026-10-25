---
title: "Microsoft Agent Framework (Rust) — Middleware"
description: "Agent, chat and function middleware, ordering and short-circuiting, common patterns, intercepting every model request, retries with RetryingChatClient, error types, and agent hooks in the community Rust port agent-framework-rs."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Middleware"
  order: 5
---

Middleware intercepts an agent's work without changing the agent: it can inspect and rewrite requests, results and tool calls, refuse them, or answer in place of the model. This page covers the three middleware levels in `agent-framework`, how they nest, what each can change, the common patterns, the client decorators that cover what middleware cannot (every model request, retries), the error types and how to match them, and the experimental agent-hooks enforcement layer. Tool behaviour itself (errors, limits, approvals) is on the [tools page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/).

Every program here was compiled against `agent-framework` %%version=agent-framework-rs%% and run. They use scripted `ChatClient`s, so they need no network and their output shows exactly when each piece runs; the [testing page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_testing/) explains the technique.

## Three levels

| Level | Builder method | Context type | Runs |
| --- | --- | --- | --- |
| Agent | `.middleware(...)` | `AgentContext` | Once per run, around everything the agent does after loading context |
| Chat | `.chat_middleware(...)` | `ChatContext` | Once per run, around the call into the chat client, which includes the whole tool loop |
| Function | `.function_middleware(...)` | `FunctionInvocationContext` | Once per local tool call |

Each builder method takes an `Arc` of the middleware and can be called repeatedly. The type aliases `AgentMiddleware`, `ChatMiddleware` and `FunctionMiddleware` in `agent_framework::middleware` name the three trait objects.

The contexts are plain structs with public fields:

| `AgentContext` | `ChatContext` | `FunctionInvocationContext` |
| --- | --- | --- |
| `messages: Vec<Message>` | `messages: Vec<Message>` | `function_name: String` |
| `is_streaming: bool` | `chat_options: ChatOptions` | `arguments: serde_json::Value` |
| `metadata: HashMap<String, Value>` | `is_streaming: bool` | `session: Option<AgentSession>` |
| `result: Option<AgentResponse>` | `metadata: HashMap<String, Value>` | `metadata: HashMap<String, Value>` (holds `call_id`) |
| `terminate: bool` | `result: Option<ChatResponse>` | `result: Option<Value>` |
| | `terminate: bool` | `terminate: bool` |
| | | `tools` (with `experimental-progressive-tools`) |

`metadata` belongs to one context and is not carried from one level to the next; to pass data from agent middleware to a tool, use the session's state, which function middleware and tools can read as `ctx.session`.

## The `Middleware` trait

A middleware implements `Middleware<C>` for its context type. Its one method, `process(&self, ctx, next)`, receives the context by value and a `Next<C>`, and returns `Result<C>`:

- to continue, call `next.run(ctx).await`, which runs the rest of the chain and returns the context with `result` filled in;
- to change the request, mutate `ctx` before calling `next.run`;
- to change the outcome, mutate the returned context's `result`;
- to answer without going further, set `ctx.result` and return `Ok(ctx)` without calling `next`;
- to fail, return an error.

`Next::run` consumes `self`, so the rest of the chain can run at most once per call: middleware cannot retry what is below it by calling `next` twice. The trait is generic over the context, so one struct can serve as middleware at every level, as `Tag` does below.

### Ordering

Middleware registered first runs outermost: its code before `next.run` runs first and its code after `next.run` runs last. The levels nest in a fixed order, which the next program prints:

```rust
@@mw_00@@
```

```text
It is 21 degrees.
agent A >
  agent B >
  chat A >
   chat B >
    model request (1 messages)
      function A >
       function B >
        tool runs
       function B <
      function A <
    model request (3 messages)
   chat B <
  chat A <
  agent B <
agent A <
run took 0 ms
```

The run made two model requests, one that asked for the tool and one that answered, but each chat middleware ran once, around both. The `EveryModelCall` wrapper, a `ChatClient` decorator, is what saw each request; see [intercepting every model request](#intercepting-every-model-request).

### Closures with `FnMiddleware`

`agent_framework::middleware::FnMiddleware::new(closure)` adapts an async closure `Fn(C, Next<C>) -> impl Future<Output = Result<C>>`, as the `timing` middleware above shows. Annotate the closure's parameter types so the context type is known, and clone any captured `Arc` into the future, as for a tool closure. A struct is clearer once the middleware has configuration or state.

`MiddlewarePipeline<C>` is the type the agent uses to run a chain; `MiddlewarePipeline::new(vec![...]).execute(ctx, terminal)` runs middleware around your own terminal function, which is handy for testing a middleware on its own.

## Agent middleware

### What `AgentContext` holds

Agent middleware runs after the agent has prepared the request. By then the context providers have run, so `ctx.messages` is the complete list about to go to the model: the instructions as a leading system message, the history loaded from the session, anything providers added, and the caller's new messages last. Its `result` is the `AgentResponse` that the run returns. After the middleware chain finishes, the context providers' `after_run` hooks record the run, and what they record is the caller's original input plus the final `result`, not the middleware's edited `messages`:

```rust
@@mw_01@@
```

```text
hello -> ["system:Be brief.", "user:hello", "system:Reply in French."] {"checked_by": String("desk")}
 ping -> pong {}
again -> ["system:Be brief.", "user:hello", "assistant:[\"system:Be brief.\", \"user:hello\", \"system:Reply in French.\"]", "user:ping", "assistant:pong", "user:again", "system:Reply in French."] {"checked_by": String("desk")}
stream: 1 update(s)
terminate: Err(AgentExecution("agent produced no result"))
```

The third turn shows both rules. The system message the middleware added on the first turn is not in the history, because it was never part of the input; the canned `pong` is, because it was the result. So agent middleware is the right place to add per-request context that should not be remembered, and the wrong place to rewrite what the user said if the rewrite must be stored (agent hooks can do the latter; see [agent hooks](#agent-hooks-experimental)). `AgentResponse::additional_properties` is a convenient place to attach your own annotations to the result.

Because agent middleware runs after history is loaded, the last message is the new input only when the caller sent one message; inspect `ctx.messages.last()` with that in mind, or compare against the roles you expect.

### Short-circuiting and `terminate`

Returning a context with `result` set and without calling `next` skips everything below: chat middleware, the model and the tools. The result is treated as the run's answer, including by the history provider, as the `pong` above shows. If the run fails instead (an error from the model or from inner middleware), the context providers' `after_run` sees the error and nothing is recorded.

`terminate` is a flag on every context. Setting it does not stop the chain by itself: the remaining middleware still run when you call `next`, but the terminal handler at the bottom of the chain sees the flag and skips its work (the model call at agent and chat level, the tool at function level). A terminated context must carry a `result`. At agent level a missing one is `Error::AgentExecution("agent produced no result")`, at chat level `"chat middleware produced no result"`, and at function level the tool's result becomes `null`. In practice, set `result` and return without calling `next`; set `terminate` too when inner middleware should run but the terminal should not.

### Streaming runs

An agent with any agent middleware does not stream. `run_stream` runs the whole middleware chain, the model and the tools to completion, then replays the response as one update per message (the `1 update(s)` above). This is the price of letting agent middleware see and change a complete result. If you need token streaming, move the logic to chat middleware or a client decorator. `ctx.is_streaming` tells middleware which entry point was used.

## Chat middleware

Chat middleware wraps the call from the agent into its chat client. It sees the final message list and the merged `ChatOptions` (per-run options already applied, tools assembled), so it is the place to enforce request settings, choose a model, add a stop sequence or answer from a cache. The run's session is available too, as `ctx.chat_options.session`, a framework-internal field that the tool loop removes before the provider sees the options.

```rust
@@mw_02@@
```

```text
  [policy] user=Some(String("ada")) streaming=false
model: temperature=Some(0.0) stop=Some(["END"])
  [policy] user=Some(String("ada")) streaming=false
(cached) model: temperature=Some(0.0) stop=Some(["END"])
  [policy] user=None streaming=true
model (streamed): temperature=Some(0.0) stop=Some(["END"])
```

On a streaming run without agent middleware, chat middleware can change the request but not the response. The chain runs before the provider stream is opened, `ctx.result` is still `None` after `next.run`, and a result set to short-circuit is ignored: the streamed answer above comes from the model even though the cache held one. This mirrors Python, whose streaming chat middleware is handed a generator rather than a response. With at least one agent middleware the run is buffered, as described above, and chat middleware gets its full powers back.

### Chat middleware runs once per run

This is the main difference from Python. Python's chat middleware wraps each model request, so in a run that calls a tool it runs once for the request that asked for the tool and again for the request that answered. In the Rust port the agent applies chat middleware outside the function-invocation loop: it runs once per run, sees the messages before the first model request, and receives the final response after the last. The `mw_00` output above shows it, as do the agent's source and the agent-hooks module documentation; the builder's doc comment and the repository's parity table still describe it as running "on every request", which is not what %%version=agent-framework-rs%% does.

The consequences: chat middleware that counts calls counts runs; one that redacts outgoing text redacts the user's input but not tool results sent back on later requests; a cache answers whole runs. To act on every model request, use a client decorator.

## Function middleware

Function middleware wraps each local tool call: approved calls, every call of a parallel batch, and calls made by the loop on behalf of a sub-agent's own tools (inside that sub-agent). It does not see hosted tools, which the provider runs, or calls to unknown tools and unparseable arguments, which the loop answers before building a context. `function_name` and `arguments` can be rewritten before `next.run`; `result` can be replaced after it; a middleware can answer without calling the tool; and `metadata["call_id"]` holds the model's call id.

```rust
@@mw_03@@
```

```text
  call0: Some(Object {"hits": Array [Object {"email": String("[email]"), "name": String("Ada")}], "limit_used": Number(50)}) None
  call1: Some(Object {"hits": Array [Object {"email": String("[email]"), "name": String("Ada")}], "limit_used": Number(50)}) None
  call2: None Some("send_email is only available 09:00-17:00 (call call2)")
search tool ran 1 time(s)
```

The order matters here. `ClampLimit` is innermost, so the cache key built by `ResultCache` contains the model's original arguments and the tool sees the clamped ones. `RedactEmails` sits outside the cache, so cached results are stored unredacted and masked on the way out. With concurrent calls (the default) both searches would have missed the cache, because they run at the same time; the example turns concurrency off to show the hit. Middleware instances are shared by all calls and all runs, so any state in them needs a `Mutex` or atomics.

How a refusal is reported decides what the model sees:

| Function middleware returns | The model receives | The run |
| --- | --- | --- |
| `Err(Error::tool_rejected(msg))` | `msg`, verbatim | continues |
| any other `Err` | `"tool execution failed"`, or the error text with `include_detailed_errors` | continues |
| `Err(Error::middleware_failure(msg))` | nothing | fails with `Error::MiddlewareFailure`; in-flight sibling calls are cancelled |
| `Ok(ctx)` with `result` set, `next` not called | that result, as if the tool had returned it | continues |

A middleware that rewrites `function_name` changes the name it and later middleware see, not which tool runs: the executor was chosen before the chain started. Function middleware is also the place to observe a tool error: match on the `Result` of `next.run(ctx).await`. The [Purview integration](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/) ships agent and chat middleware built on these same traits.

## Agent-level, not per-run

All middleware is attached when the agent is built. `AgentRunOptions` carries chat options and extra tools but no middleware, so there is no counterpart of Python's `agent.run(..., middleware=[...])`. Two ways to vary middleware per request:

- Build several agents. Building is cheap and an `Agent` is `Clone`; share the client (clone it, or wrap it in an `Arc`) and the stateful middleware instances between them.
- Make the middleware read a per-request switch from the session's state. Function middleware sees it as `ctx.session`, chat middleware as `ctx.chat_options.session`. Agent middleware has no session handle, so a switch it needs must travel in the messages, or in state that the middleware holds itself.

## Intercepting every model request

Because chat middleware runs once per run, anything that must happen on every model request is a `ChatClient` decorator: a struct that holds the provider client, implements `ChatClient` by delegating to it, and does its work around the delegated call. Hand the decorated client to `Agent::builder`. The builder puts the tool loop above it, so `get_response` runs once per model request, including the requests that carry tool results. `RetryingChatClient` and `ObservableChatClient` are built this way.

```rust
@@mw_05@@
```

```text
  request 1: 1 message(s)
  request 2: 3 message(s)
  request 3: 5 message(s)
done; I saw: card [card] was declined
  request 1: 1 message(s)
  request 2: 3 message(s)
Err(Other("model request budget of 2 spent"))
```

Forward `model()` and `service_url()` so the builder still picks up the default model and traces still name the endpoint, and delegate `get_streaming_response` (or decorate it too). An error from a decorator is a model error, not a tool error: the loop does not absorb it, and the run fails. A decorator sees what goes over the wire, after instructions are turned into a system message and history is loaded, which makes it the right layer for redaction that must also cover tool results, per-request quotas, request logging and request-level caching.

## Common patterns

Each pattern below is a small middleware; most of them appear in the program at the end of this section, which runs six questions concurrently through one agent.

### Logging and timing

Wrap `next.run` and record what came back. At agent level, log the question, the outcome and the duration; at function level, the tool name and arguments; per model request, use a decorator. The agent already emits OpenTelemetry spans (`invoke_agent`, `execute_tool`, and `chat` through `ObservableChatClient`), and a `tracing` event emitted inside middleware is attached to the current span, as the output below shows, so prefer `tracing` over printing (see the [observability page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_observability/)).

### Rate limits and concurrency

Hold a permit around `next.run`: a `tokio::sync::Semaphore` caps concurrent runs, and a per-user counter or token bucket in the middleware enforces quotas. A semaphore in agent middleware limits runs; to limit model requests, put it in a client decorator. For provider rate limits (HTTP 429), retry instead: see [retries](#retries).

### Guardrails

Input checks belong in agent middleware before `next.run`: refuse with a canned `result`. Output checks go after `next.run`: inspect `result` and replace it. Checks on actions belong in function middleware, returning `Error::tool_rejected` when the model should hear why or `Error::middleware_failure` when the run must stop. A guardrail placed outermost sees everything, including what an inner cache served. For enforcement that must not be bypassed, consider [agent hooks](#agent-hooks-experimental), which fail closed by design.

### Redaction

Redact outgoing user text in agent or chat middleware, tool results in function middleware (the `RedactEmails` example), and everything sent to the model, including tool results on later requests, in a client decorator. Remember that agent-level edits to `messages` are not stored, while a replaced `result` is.

### Caching

A cache at agent level answers whole runs from the input; at chat level, model calls from the message list and options; at function level, tool calls from name and arguments. Key on everything that affects the answer: at agent level that is all of `ctx.messages`, history included, not just the last message. Store cached responses without `usage_details`, so that token accounting does not count tokens that were never spent.

### Fallbacks and usage tracking

To degrade gracefully, match on the result of `next.run` and build a replacement context. `next.run` consumes the context, so keep what you need (the messages, `is_streaming`) beforehand and rebuild with `AgentContext::new(messages, is_streaming)`. For usage, read `result.usage_details` after `next.run`; the tool loop sums usage over every model request of the run, so one read covers the whole run.

```rust
@@mw_04@@
```

```text
 INFO invoke_agent{gen_ai.operation.name="invoke_agent" gen_ai.agent.name="..." gen_ai.agent.id="..." otel.name="invoke_agent ..."}: agent run question="config?" status="ok" ms=50
 INFO invoke_agent{...}: agent run question="c?" status="ok" ms=50
 INFO invoke_agent{...}: agent run question="b?" status="ok" ms=102
 INFO invoke_agent{...}: agent run question="a?" status="ok" ms=102
Ok("answer to a?")
Ok("answer to b?")
Ok("answer to c?")
 INFO invoke_agent{...}: agent run question="d?" status="ok" ms=153
 INFO invoke_agent{...}: agent run question="outage" status="ok" ms=153
Ok("answer to d?")
Ok("[answer withheld: it contained a credential]")
Ok("The service is busy; please try again shortly.")
peak concurrent runs: 2
```

The order of registration is the order of reasoning: `LogRuns` is outermost, so it logs what the caller receives (`status="ok"` even for the outage, because `Fallback` inside it turned the error into an answer); the limiter sits outside the guardrail and fallback so that their work counts as part of a run; `Fallback` is innermost, nearest the model. The durations show the semaphore at work: two runs at a time, in waves of about 50 ms.

## Retries

### `RetryPolicy`

`RetryingChatClient::new(client).with_policy(policy)` wraps any client and re-sends a request that failed with a retryable error. `RetryPolicy` has public fields and builder methods:

| Field (builder) | Default | Meaning |
| --- | --- | --- |
| `max_retries` (`with_max_retries`) | 3 | Retries after the first attempt, so up to four attempts |
| `initial_delay` | 500 ms | Delay before the first retry |
| `backoff_multiplier` | 2.0 | Growth per retry |
| `max_delay` | 30 s | Cap on any one delay, including a server's `Retry-After` |
| `jitter` | 0.3 | The computed delay is multiplied by `1 - jitter * r` for a random `r` in `[0, 1)`; `0.0` makes delays exact. The builder clamps it to `0.0..=1.0`, a struct literal does not |
| `retry_on` | `RetryOn::Default` | Which errors to retry |

The delay before retry *n* is `initial_delay * backoff_multiplier^(n-1)`, capped at `max_delay`, then jittered downwards. When the error carries a `Retry-After` (an `Error::ServiceStatus` with `retry_after: Some(seconds)`), that value is used instead, capped at `max_delay` and not jittered. The OpenAI, Azure OpenAI and Anthropic clients parse the header into the error. Each retry is logged as a `tracing` warning with the attempt, delay, status and error.

`RetryOn::Default` retries `Error::ServiceStatus` with status 408, 429 or 5xx, and an `Error::Service` whose message mentions a transport failure (`request failed`, `timed out`, `timeout`, `connection`, `stream error`). It never retries `ServiceInvalidAuth`, `ServiceInvalidRequest` or `ServiceContentFilter`, or anything that is not a service error. `RetryOn::predicate(|err| ...)` replaces the rule entirely.

```rust
@@mw_06@@
```

```text
503, 503, 503, then success (third delay capped at 300 ms):
  t=   0 ms  service error: unavailable
  t= 101 ms  service error: unavailable
  t= 303 ms  service error: unavailable
  t= 605 ms  ok
  -> hello
429 with Retry-After: 0.05 s wins over the computed 100 ms:
  t=   0 ms  service error: slow down
  t=  51 ms  ok
  -> hello
401 is never retried:
  t=   0 ms  service error: 401 bad key
  -> Err(ServiceInvalidAuth { message: "401 bad key" })
retries exhausted: the last error is returned:
  t=   0 ms  service error: request failed: connection reset
  t= 101 ms  service error: request failed: connection reset
  -> Err(Service("request failed: connection reset"))
a custom rule: also retry content-filter refusals once:
  t=   0 ms  service error: blocked
  t= 101 ms  ok
  -> hello
```

When the retries run out, the last error is returned unchanged, so the code that handles errors does not need to know a retry layer exists. For streaming, only opening the stream is retried: a failure establishing it, or in its first item before anything reached the consumer, is retried; once an update has been delivered, later stream errors are passed through, because the consumer has already seen partial output.

### Where the retry layer goes

Wrap the provider client before handing it to `Agent::builder`. The retry layer then sits below the tool loop, so each model request is retried on its own: a 503 on the request that carries tool results re-sends that request, and the tools do not run again. Wrapping a whole agent run in a retry loop of your own would re-run tools that already ran.

With `ObservableChatClient` as well, the order decides what a span covers. `ObservableChatClient::new(RetryingChatClient::new(client), "openai")` records one `chat` span per logical request, retries included; `RetryingChatClient::new(ObservableChatClient::new(client, "openai"))` records a span per attempt, which shows the failed attempts in traces. Custom decorators compose the same way, outermost first.

Python's agent framework documents retries as a middleware pattern rather than shipping a built-in layer; this port has `RetryingChatClient` built in.

### Retrying tools

Tool errors are not retried by `RetryingChatClient`, which only sees model requests. The model already gets a second chance (it sees the failure and can call again, up to `max_consecutive_errors_per_request`), which is often enough. For transparent retries, put the retry inside the tool: loop in the closure, or wrap the tool in a `Tool` implementation that calls the inner tool's `invoke_in_context` several times. Function middleware cannot do it, because `Next::run` can only be called once.

## Errors

Every fallible call returns `agent_framework::Result<T>`, an alias for `Result<T, Error>`. `Error` is a `thiserror` enum, marked `#[non_exhaustive]`, whose variants mirror Python's exception hierarchy:

| Variant | Raised for |
| --- | --- |
| `ServiceStatus { status, message, retry_after }` | A non-success HTTP status not classified more precisely, notably 408, 429 and 5xx |
| `ServiceInvalidAuth { message }` | Rejected credentials (401, 403) |
| `ServiceInvalidRequest { message }` | A malformed or invalid request (400, 404, 422) |
| `ServiceContentFilter { message, detail }` | A content-filter refusal, with the provider's breakdown when it sent one |
| `Service(String)` | Transport failures, stream decoding errors, error payloads in a 200 response |
| `Tool(String)` | A failed tool; also an unknown tool with `terminate_on_unknown_calls` |
| `ToolRejected(String)` | A refusal addressed to the model (absorbed by the tool loop, never surfaces from a run) |
| `MiddlewareFailure(String)` | Fail-closed refusal from function middleware or a tool; ends the run |
| `InterceptionBlocked(..)` | An agent-hooks deny (with `experimental-agent-hooks`) |
| `AgentExecution(String)` | Run-level problems, such as middleware that produced no result or a service-managed session that lost its conversation id |
| `AgentInitialization(String)`, `Configuration(String)` | Construction problems, such as a missing environment variable in `from_env` or an invalid `FunctionInvocationConfig` |
| `Workflow(String)` | Workflow validation and execution |
| `Serialization(String)`, `Json(serde_json::Error)`, `Content(String)`, `AdditionItemMismatch(String)` | Data conversion |
| `Other(String)` | Anything else; `Error::other(x)` wraps any `Display` |

Constructor helpers exist for the common ones: `Error::service`, `service_status`, `service_invalid_auth`, `service_invalid_request`, `service_content_filter`, `service_content_filter_with_detail`, `tool`, `tool_rejected`, `middleware_failure` and `other`. Accessors read details without a match: `status()` and `retry_after()` (both `Some` only for `ServiceStatus`), `content_filter_detail()`, `is_middleware_failure()` and, with agent hooks, `interception_blocked()`. The three classified service variants deliberately carry no status code (it is in the message), so `status()` is `None` for them.

### Matching errors

```rust
@@mw_07@@
```

```text
rate limited; retry after Some(2.0) s                   status=Some(429) retry_after=Some(2.0)
provider error 502: Bad Gateway                         status=Some(502) retry_after=None
check the API key                                       status=None retry_after=None
bad request: OpenAI API error 400: unknown parameter    status=None retry_after=None
filtered (Some("ResponsibleAIPolicyViolation"), param Some("prompt")): ["violence"] status=None retry_after=None
transport: request failed: connection refused           status=None retry_after=None
blocked by policy: tenant suspended                     status=None retry_after=None
misconfigured: OPENAI_API_KEY is not set
```

A wildcard arm is required because the enum is non-exhaustive, which lets the crate add variants in a minor release. `ContentFilterDetail` (in `agent_framework::error`) has the provider's `code` (compare with `ContentFilterDetail::RESPONSIBLE_AI_POLICY_VIOLATION` or `CONTENT_FILTERED`), the `param` that tripped it (`"prompt"` for input, absent for a filtered completion), and per-category `ContentFilterCategory` values with `filtered`, `detected` and `severity`; `filtered_categories()` lists the ones that fired. Every field is an open string rather than an enum, so a new Azure category arrives as itself. The `Display` text of all service variants starts with `service error:`.

Not every failure is an error. A response cut off by the token limit or a content filter in the middle of generation can still succeed; check `response.finish_reason` (`FinishReason::LENGTH`, `FinishReason::CONTENT_FILTER`) as well. And a tool failure never surfaces from a run unless it is a `MiddlewareFailure`: the model sees it instead.

## Agent hooks (experimental)

The `experimental-agent-hooks` feature implements the [AGENT-HOOKS 0.1](https://github.com/responsibleai/agent-hooks) control contract: a fixed set of interception points, a JSON context handed to interceptors at each, and an `allow`, `deny` or `transform` verdict that the framework must obey. It is the Rust counterpart of Python's `create_agent_hooks_middleware` and .NET's `AgentHooksAgent`. Where ordinary middleware is a toolkit, hooks are an enforcement layer: they fail closed when anything goes wrong, and they are installed as a unit so that no seam can be left out.

```rust
@@mw_08@@
```

```text
[customer details withheld]
  agent_startup   Allow None
  input           Allow None
  pre_model_call  Allow None
  post_model_call Allow None
  pre_tool_call   Transform None
  post_tool_call  Allow None
  pre_model_call  Allow None
  post_model_call Allow None
  output          Transform None
  agent_shutdown  Allow None
blocked at input because Some("credential_request"): blocked by agent-hooks at input (credential_request)
Customer record: {"card":"4111 1111 1111 1111","id":42}
Some(Configuration("agent-hooks enforcement requires at least one interceptor (an emitter with none fails closed on every emission); register an explicit allow-all interceptor for a deliberate passthrough"))
```

`AgentHooks::new(options)?.build_agent(builder)`, or `builder.build_with_agent_hooks(hooks)`, installs every seam and returns an `AgentHooksAgent`. It implements `SupportsAgentRun` (`run`, `run_with_options`, `run_stream`, `create_session`) but not the `Agent`-only conveniences such as `run_once`. Pass the raw provider client to the builder: the model-call seam wraps it, and a client that already runs its own tool loop would execute tools outside the tool seam.

### Interception points

| Point | When | `target` | Seam |
| --- | --- | --- | --- |
| `agent_startup` | once per run, first | `{"tools_registered": [...]}` | the `AgentHooksAgent` wrapper |
| `input` | before the run, on the caller's messages only | `{"content", "role"}` | the wrapper |
| `pre_model_call` | before every model request | the message list | a client decorator below the tool loop |
| `post_model_call` | after every model response | `{"content", "tool_calls", "finish_reason"}` | the same decorator |
| `pre_tool_call` | before every local tool call | the arguments object | function middleware, innermost |
| `post_tool_call` | after every local tool call | the result value | function middleware, outermost |
| `output` | before the response is returned and stored | `{"content"}` | agent middleware, outermost |
| `agent_shutdown` | once per run, last | `{"reason"}` | the wrapper |

Unlike chat middleware, `pre_model_call` and `post_model_call` bracket each model request, as the two pairs in the output show. An interceptor receives an `InterceptionContext`: `point()`, `sequence()`, `session_id()`, `target()` (the value a transform rewrites), `get(field)` for the other fields (`tool_call`, `model`, `usage`, ...), and `as_json()` for the whole wire object. Plain text is projected as a string and anything richer as content objects, never flattened.

### Verdicts and transforms

An interceptor implements the async `Interceptor` trait, or is built from a closure with `interceptor_fn`. It returns a `Verdict`:

- `Verdict::allow()`; `Verdict::warn(reason, message)` allows and records a warning.
- `Verdict::deny(reason)`, optionally `.with_message(...)`. `Verdict::escalate(reason)` is a deny marked as liftable by an approval step, which this port does not have, so it is enforced as a plain deny.
- `Verdict::transform(path, value)` replaces the value at a path rooted at `$target` (dot members, `[index]` and `["member"]`): `$target.include_card` in the tool arguments, `$target.content` in the output. Transforms are not allowed at `agent_startup` and `agent_shutdown`.
- `Verdict::from_json(value)` parses a verdict from a remote policy service; unknown fields are rejected.

A deny at `agent_startup`, `input`, `pre_model_call`, `post_model_call` or `output` fails the run with `Error::InterceptionBlocked`, whose `interception_blocked()` gives the point, reason and payload-free record. A deny at `pre_tool_call` stops that call (and `post_tool_call` discards its result); the model gets a tool error naming the reason and the run continues. A transform is written back into the native value, so the tool receives the rewritten arguments and history stores the rewritten output; a transformed output drops the response's parsed `value`. Interceptors run in registration order and the first deny wins (the `sequential/first_deny` profile).

### Enforcement, modes and records

Failures are denies with reserved `host_error:*` reasons (`agent_framework::agent_hooks::host_error`): an interceptor that errors, panics or times out, an invalid verdict, a transform that cannot be applied, a context over the size bounds. At the tool seam such a failure stops the whole run through `Error::MiddlewareFailure`. Streaming is fail-closed by buffering: model streams are assembled before `post_model_call`, and the agent's stream is released only after `output`. A denied run persists nothing.

`AgentHooksOptions` configures the bundle: `interceptor(...)` and `named_interceptor(name, ...)` register interceptors (at least one is required), `timeout(Some(duration))` sets the per-interceptor timeout (default 5 seconds, `DEFAULT_TIMEOUT`; `None` disables it), `mode(EnforcementMode::EvaluateOnly)` records verdicts without enforcing them (the last run in the example returned the card number despite the deny), and `record_sink(...)` receives an `InterceptionRecord` for every emission, with the verdict, the deciding interceptor and per-interceptor summaries but no payload. `AgentHooks::from_emitter(emitter, builder)` binds the bundle to a host-owned `InterceptionEmitter` and `InterceptionContextBuilder`, for hosts that run one agent-hooks session across several agent runs and emit `agent_startup` and `agent_shutdown` themselves.

### What is not ported

There is no approval seam, so liftable denies are plain denies. Only the `sequential/first_deny` composition profile exists. Result labels are recorded but not carried to later points. Records carry no identity hashes (`identity_provider` is `null`). Dropping a run's future emits no `agent_shutdown`. Hosted tools never pass the tool seam; their calls and results are visible in `post_model_call`. A plain sub-agent used as a tool is covered only at the tool seam that calls it. The feature can change in any minor release; the module documentation on docs.rs lists the specification sections it implements.

## Differences from Python

- **Chat middleware** runs once per run, around the whole tool loop, instead of once per model request (see [above](#chat-middleware-runs-once-per-run)). Use a `ChatClient` decorator for per-request work.
- **Registration.** Middleware is attached to the agent only; there is no per-run `middleware=` argument, no decorator form (`@agent_middleware`), and no `MiddlewareBundle`. `FnMiddleware` is the closure form.
- **Termination.** There is no `MiddlewareTermination` exception: set `result` and return without calling `next`. `terminate` only tells the terminal handler to skip its work.
- **Agent middleware.** It runs after context providers have loaded history, so it sees the full message list, and its edits to `messages` are not stored. With agent middleware, `run_stream` is buffered.
- **Streaming hooks.** Python exposes streaming hooks on `ChatContext` and `AgentContext`; this port has none, and chat middleware on an unbuffered stream can only change the request.
- **Not ported.** Message injection into a running session (`MessageInjectionMiddleware`) is not ported.
- **Retries.** These are built in as `RetryingChatClient`; Python shows them as a middleware pattern.

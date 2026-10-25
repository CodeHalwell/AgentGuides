---
title: "Microsoft Agent Framework (Rust) — MCP"
description: "Model Context Protocol in agent-framework-rs: stdio, streamable HTTP and WebSocket servers as tool sources, tool filtering and approval, result mapping, prompts, sampling, roots, authentication headers, the low-level McpClient, lifecycle and reconnecting, and serving an agent to MCP clients."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "MCP"
  order: 10
---

The `mcp` feature adds a Model Context Protocol client, `agent_framework::mcp`. It connects to a server, lists its tools and turns each into an ordinary `ToolDefinition` whose executor forwards the call to the server. Everything above that point (the tool loop, approvals, middleware, telemetry) treats an MCP tool exactly like a local [function tool](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/). The crate also answers the server's own requests for sampling and roots, and exposes prompts.

The examples on this page were compiled against `agent-framework` %%version=agent-framework-rs%% and run: the stdio and HTTP ones against the reference server `@modelcontextprotocol/server-everything` (started with `npx`, over stdio and over its streamable HTTP transport), with a stand-in for the OpenAI API; the rest against in-process servers. The WebSocket transport was compiled only.

```toml
[dependencies]
agent-framework = { version = "%%version=agent-framework-rs%%", features = ["mcp"] }
tokio = { version = "1", features = ["full"] }
serde_json = "1"
```

The prelude brings in `McpStdioTool`, `McpStreamableHttpTool` and `McpWebsocketTool`; everything else is in `agent_framework::mcp`.

## Choosing a transport

| Type | Server | Use it for |
| --- | --- | --- |
| `McpStdioTool` | A child process you spawn; newline-delimited JSON-RPC over its stdin and stdout | Local servers, `npx`/`uvx` packages, development |
| `McpStreamableHttpTool` | An HTTP endpoint; each message is a POST, answered with JSON or an SSE stream | Shared or remote servers, anything behind a gateway that checks a header |
| `McpWebsocketTool` | A `ws://` or `wss://` URL with the `mcp` subprotocol, one JSON-RPC message per text frame | Servers that push notifications or sampling requests outside a call |

All three have the same builder methods for tools, prompts, approval, sampling and roots, and all three implement `ToolSource`, so swapping one for another is a change to the constructor. The client speaks protocol revision `2025-06-18` (`PROTOCOL_VERSION`) and accepts a server that negotiates down to `2025-03-26` or `2024-11-05` (`COMPATIBLE_PROTOCOL_VERSIONS`); any other version is accepted with a warning in the log rather than refused. Newer revisions are not negotiated.

## Connecting a stdio server

`McpStdioTool::new(name, command)` describes the server; nothing is spawned until the first call that needs a connection. Registering it with `tool_source` makes the agent connect on its first run:

```rust
use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let mcp = Arc::new(
        McpStdioTool::new("everything", "npx")
            .args(["-y", "@modelcontextprotocol/server-everything"])
            .description("The MCP reference server")
            // npx downloads the package, so it needs the proxy and cache
            // variables; this passes the whole environment through.
            .inherit_parent_environment(true)
            // Only these two tools are offered to the model.
            .allowed_tools(["echo", "get-sum"]),
    );

    // Optional: list what the agent will see. This connects (spawning the
    // process and running the initialize handshake) if it has not already.
    for tool in mcp.tool_definitions().await? {
        println!("{}: {}", tool.name, tool.description);
    }

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("Use the tools when they help.")
        .tool_source(mcp.clone())
        .build();
    println!("{}", agent.run_once("Echo back: hello from MCP").await?.text());

    mcp.close().await?;
    Ok(())
}
```

`connect()` spawns the process and runs the `initialize` handshake and the `notifications/initialized` notification. It is idempotent and safe to call concurrently: the first caller does the work and the others wait for its result. `tool_definitions()`, `prompts()`, `get_prompt()` and every tool call connect first if needed, so an explicit `connect()` is only for failing early, at startup rather than on the first run.

| Method | Default | Effect |
| --- | --- | --- |
| `args([...])` | none | Command-line arguments |
| `cwd(path)` | the parent's | Working directory of the server process |
| `env([(k, v)])` | none | Explicit variables, applied last (they win) |
| `inherit_env_var(name)`, `inherit_env_vars([...])` | none | Copy named variables from the parent |
| `inherit_parent_environment(bool)` | `false` | Pass the whole parent environment |
| `request_timeout(duration)` | none (wait for ever) | Per-request timeout for every JSON-RPC call, the handshake included |
| `description(text)` | none | Read back with `description_text()` |
| `allowed_tools([...])`, `approval_mode(...)`, `load_tools(bool)`, `load_prompts(bool)` | all tools, no approval, `true`, `true` | See [filtering and approval](#filtering-naming-and-approval) |
| `sampling_handler(handler)`, `roots([...])` | none | See [sampling and roots](#sampling-and-roots) |

### The child's environment

A stdio server does not inherit the parent's environment. It starts from a short allowlist (`PATH`, `HOME`, `TMPDIR`, `TMP`, `TEMP`, `LANG`, `LC_ALL`, `LC_CTYPE`, and the Windows variables a process needs to start, such as `SystemRoot`, `USERPROFILE`, `APPDATA` and `PATHEXT`), adds the variables named with `inherit_env_var`, then the explicit `env` pairs. This keeps API keys, cloud credentials and tokens in the host's environment away from a package that `npx` has just downloaded. `inherit_parent_environment(true)` restores full inheritance, and with it the explicit pairs are layered on top.

Proxy, certificate and npm cache settings are not in the allowlist. A launcher that downloads its package (`npx -y`, `uvx`) behind a proxy then cannot fetch it, and, with no `request_timeout`, the `initialize` request waits for ever. Either pass the variables it needs, for example `.inherit_env_vars(["HTTPS_PROXY", "NO_PROXY", "NODE_EXTRA_CA_CERTS"])`, or inherit everything as the example does; and set a timeout, so that a server that never answers fails the run with `MCP request 'initialize' timed out after ...` instead of hanging it.

The server's stderr is drained to `tracing` at debug level under the target `mcp::server_stderr`, so `RUST_LOG=mcp::server_stderr=debug` shows it. A non-JSON line on stdout is logged and skipped.

## Tool sources and snapshots

There are two ways to give a server's tools to an agent, and they differ in when the tool list is read:

- **`.tool_source(Arc::new(mcp))`** resolves the tools on every `run` and `run_stream`. The first resolution connects and calls `tools/list` (following `nextCursor` pagination); later ones are served from a cache in the client, which is emptied when the server sends `notifications/tools/list_changed`. A connection or listing error fails the run rather than leaving the agent silently without tools. `load_tools(false)` makes resolution return nothing without connecting.
- **`.tools(mcp.tool_definitions().await?)`** lists the tools once, live (no cache, and regardless of `load_tools`), and freezes them into the agent. The definitions keep working, since each executor holds the shared session, but the agent never sees a tool added later.

Use a tool source for a server whose catalogue changes or that you want to start lazily, and a snapshot when you want the agent's tool list fixed at build time, for example so that a test can assert on it. Tools from sources are appended after the agent's own, per-run and context-provider tools. The cache is per client, so several agents sharing one `Arc<McpStdioTool>` share one session, one process and one cache.

Over streamable HTTP, a `list_changed` notification is only seen when it arrives inside the SSE response to a request that is in flight, because the transport does not open the standalone GET stream a server uses to push messages between requests. A WebSocket or stdio server's notifications are always seen.

## Filtering, naming and approval

Tool names are normalised before the model sees them: any character outside `A-Za-z0-9_.-` becomes `-`, so `weather/get current` is offered as `weather-get-current` (calls still go to the server under the original name). `allowed_tools` takes the normalised names and keeps the server's order. If two tools normalise to the same name, the first is kept and the second skipped with a warning; prompts are de-duplicated the same way.

`approval_mode` sets `ApprovalMode` on each produced definition:

| `McpApprovalMode` | Effect |
| --- | --- |
| `never_require_all()` | No tool needs approval (the default) |
| `always_require_all()` | Every tool needs approval |
| `per_tool(["name", ...])` | Only the named (normalised) tools need approval |

A tool that needs approval pauses the run exactly as a local one does, with a `FunctionApprovalRequestContent` returned by `user_input_requests()`; the answer goes back on the same session. The [tool approvals](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_comprehensive_guide/#tool-approvals) section explains the round trip. This example takes a snapshot, so the approval flags can be printed before the run:

```rust
use agent_framework::mcp::McpApprovalMode;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let mcp = McpStdioTool::new("everything", "npx")
        .args(["-y", "@modelcontextprotocol/server-everything"])
        .inherit_parent_environment(true)
        .allowed_tools(["get-sum", "get-tiny-image"])
        // Only get-sum pauses for a human; the other tool runs straight away.
        .approval_mode(McpApprovalMode::per_tool(["get-sum"]));

    // A snapshot: these definitions never change, whatever the server does later.
    let tools = mcp.tool_definitions().await?;
    for tool in &tools {
        println!("{} (approval: {})", tool.name, tool.requires_approval());
    }

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("Use the tools for arithmetic.")
        .tools(tools)
        .build();

    let mut session = agent.create_session();
    let mut input = vec![Message::user("What is 3 + 4?")];
    for _ in 0..5 {
        let response = agent.run(input, Some(&mut session)).await?;
        let requests = response.user_input_requests();
        if requests.is_empty() {
            println!("{}", response.text());
            break;
        }
        let mut answers = Vec::new();
        for request in requests {
            println!("approve {}({:?})? yes", request.function_call.name, request.function_call.arguments);
            answers.push(Content::FunctionApprovalResponse(request.create_response(true)));
        }
        input = vec![Message::with_contents(Role::user(), answers)];
    }

    mcp.close().await?;
    Ok(())
}
```

Python's `approval_mode` mapping has both an `always_require_approval` and a `never_require_approval` list, because its approval setting can be unset. Rust's `ApprovalMode` has no unset state, so `per_tool` needs only the list of tools that require approval. This `McpApprovalMode` is not the core `agent_framework::tools::McpApprovalMode` (`Always`, `Never`, `PerTool`), which configures [hosted MCP](#hosted-mcp) tools that the model provider calls; the two are easy to confuse because both are called `McpApprovalMode`.

## Tool results

A `tools/call` result is a list of content blocks plus an `isError` flag and optional `structuredContent`. The executor (`McpClient::call_tool_value`) turns it into the single JSON value the model sees:

| Result | Value handed to the model |
| --- | --- |
| One text block that is valid JSON | The parsed JSON |
| One text block | A JSON string |
| No content, with `structuredContent` | The structured content |
| No content | `null` |
| Several blocks, or a non-text block | An array of the blocks as JSON (`type`, `text`, `data`, `mimeType`, ...) |
| `isError: true` | `Err(Error::Tool(text))`, which the tool loop reports to the model as a failed call |

`McpClient::call_tool` returns the raw `CallToolResult` (`content: Vec<ContentBlock>`, `is_error`, `structured_content`) instead. `ContentBlock` has `Text`, `Image`, `Audio`, `Resource`, `ResourceLink` and `Unknown` variants; `to_core_content()` converts one to a core `Content` (images and audio become `data:` URIs, a resource link a `Content::Uri`), and an unrecognised block is kept as `Unknown` rather than dropped. Note that a multi-block result, such as an image with a caption, reaches the model as JSON text with the base64 payload inside, not as an image the model can see.

## Streamable HTTP and WebSocket

`McpStreamableHttpTool::new(name, url)` takes `headers([...])` (sent on every request) and `timeout(duration)`; `McpWebsocketTool::new(name, url)` takes `headers` for the upgrade request and `request_timeout`. The example connects to the reference server's HTTP transport, first through the tool, then through the low-level client to show what the transport does:

```rust
use std::sync::Arc;
use std::time::Duration;

use agent_framework::mcp::{ContentBlock, McpClient, McpStreamableHttpTransport};
use agent_framework::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let url = std::env::var("MCP_URL").unwrap_or_else(|_| "http://127.0.0.1:3001/mcp".into());
    let token = std::env::var("MCP_TOKEN").unwrap_or_default();

    // The high-level tool: the same builder as stdio, plus headers and a timeout.
    let mcp = Arc::new(
        McpStreamableHttpTool::new("everything", &url)
            .headers([("Authorization", format!("Bearer {token}"))])
            .timeout(Duration::from_secs(30))
            .allowed_tools(["echo"]),
    );
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .tool_source(mcp.clone())
        .build();
    println!("{}", agent.run_once("Echo: over HTTP").await?.text());
    mcp.close().await?; // DELETEs the MCP session on the server

    // The low-level client: own the transport, see the raw protocol results.
    let headers = McpStreamableHttpTransport::header_map(&[("Authorization".into(), format!("Bearer {token}"))])?;
    let transport = Arc::new(McpStreamableHttpTransport::new(&url, headers, Some(Duration::from_secs(30)))?);
    let client = McpClient::new(transport.clone());
    let init = client.initialize("my-app", "1.0.0").await?;
    println!("server {} {} speaks {}", init.server_info.name, init.server_info.version, init.protocol_version);
    println!("session id: {:?}", transport.session_id().await);
    client.ping().await?;
    println!("{} tools, prompts supported: {}", client.list_tools().await?.len(), client.supports_prompts().await);

    // call_tool keeps isError and every content block; call_tool_value is
    // what the agent's tool executor uses.
    let result = client.call_tool("get-structured-content", json!({ "location": "Chicago" })).await?;
    println!("is_error={} blocks={} structured={:?}", result.is_error, result.content.len(), result.structured_content);
    let image = client.call_tool("get-tiny-image", json!({})).await?;
    for block in &image.content {
        match block {
            ContentBlock::Text(text) => println!("text: {text}"),
            ContentBlock::Image { mime_type, data } => println!("image: {mime_type}, {} base64 chars", data.len()),
            other => println!("other: {other:?}"),
        }
    }
    client.close().await?;
    Ok(())
}
```

The HTTP transport:

- POSTs each message with `Accept: application/json, text/event-stream`, and reads either a JSON body or an SSE stream scanned for the response with the matching id. A server request embedded in that stream (sampling, roots, ping) is answered with a follow-up POST.
- Captures the `Mcp-Session-Id` header and sends it on every later request; `close()` sends a best-effort `DELETE` for the session.
- Follows redirects itself, up to ten, and sends the configured headers and the session id only while the target has the same scheme, host and port as the configured URL. A redirect elsewhere gets neither, so an `X-Api-Key` or bearer token cannot leak to another host. Method and body are kept on every hop.
- Rejects a URL that is not absolute `http` or `https` with a configuration error.

A non-2xx response becomes `Error::Service` with the status and body. There is no OAuth flow: obtain a token yourself and pass it as a header. Headers are fixed when the tool is built, so a short-lived token means building a new tool when it expires. Python's `header_provider` callback, which supplies headers per request, has no counterpart.

The WebSocket transport has the same builder surface. Prefer it to HTTP when the server sends requests or notifications outside a call, since the socket stays open in both directions.

## Prompts

`prompts()` lists the server's prompts (`PromptDescriptor` with `name`, `description` and `arguments`), cached like tools until `notifications/prompts/list_changed`. It returns an empty list without a request when `load_prompts(false)` is set or the server did not declare the `prompts` capability. `get_prompt(name, args)` renders one as core `Message`s, ready to pass to `agent.run`; arguments are a JSON object of strings, as the protocol defines them, and `Value::Null` is sent as `{}`.

```rust
use agent_framework::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let mcp = McpStdioTool::new("everything", "npx")
        .args(["-y", "@modelcontextprotocol/server-everything"])
        .inherit_parent_environment(true)
        // This agent uses the server for prompts only.
        .load_tools(false);

    for prompt in mcp.prompts().await? {
        let args: Vec<String> = prompt
            .arguments
            .iter()
            .flatten()
            .map(|a| format!("{}{}", a.name, if a.required == Some(true) { "*" } else { "" }))
            .collect();
        println!("{}({}): {}", prompt.name, args.join(", "), prompt.description.unwrap_or_default());
    }

    // Arguments are a JSON object of strings, as the MCP spec defines them.
    let messages = mcp
        .get_prompt("args-prompt", json!({ "city": "Bath", "state": "Somerset" }))
        .await?;
    for message in &messages {
        println!("[{}] {}", message.role, message.text());
    }

    // The rendered prompt is ordinary messages: run it like any other input.
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?).build();
    println!("{}", agent.run(messages, None).await?.text());

    mcp.close().await?;
    Ok(())
}
```

`get_prompt` does not check the capability first: asking a server without prompts gives the server's JSON-RPC error as `Error::Service`. Prompts are not turned into tools or added to the agent automatically.

## Sampling and roots

A server can call back into the client. Two such requests are supported:

- **Sampling** (`sampling/createMessage`): the server asks the client's model for a completion. Register a `SamplingHandler`, an `Arc<dyn Fn(CreateMessageParams) -> BoxFuture<Result<CreateMessageResult>>>`; `chat_client_sampling_handler(client)` adapts any `ChatClient`. It maps the request's messages, `systemPrompt`, `maxTokens`, `temperature` and `stopSequences` onto a `get_response` call and returns the first text (or image or audio) content of the reply, with the model name from the response. `modelPreferences`, `includeContext` and `metadata` are parsed but not acted on, and the tool-use fields of later protocol revisions are not modelled.
- **Roots** (`roots/list`): the server asks which locations it should treat as the user's workspace. `roots([Root::new(uri).with_name(name)])` sets a static list.

Each capability is declared during `initialize` only when its handler is set, so configure them before the first connection. The handlers do not go through the agent: a sampling request runs on whatever client you gave the handler, outside the agent's middleware and tool loop. Here both are exercised by calling the reference server's demonstration tools directly, without a model in between:

```rust
use std::sync::Arc;

use agent_framework::mcp::{chat_client_sampling_handler, Root};
use agent_framework::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    // The model that answers the server's sampling requests. It need not be
    // the agent's model; here it is a smaller one.
    let sampler: Arc<dyn ChatClient> =
        Arc::new(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?);

    let mcp = McpStdioTool::new("everything", "npx")
        .args(["-y", "@modelcontextprotocol/server-everything"])
        .inherit_parent_environment(true)
        // Declares the `sampling` capability during initialize.
        .sampling_handler(chat_client_sampling_handler(sampler))
        // Declares the `roots` capability and answers roots/list.
        .roots([
            Root::new("file:///workspace").with_name("Workspace"),
            Root::new("file:///tmp").with_name("Scratch"),
        ])
        .allowed_tools(["trigger-sampling-request", "get-roots-list"]);

    // Call the tools directly, without a model in the loop: each definition
    // carries an executor that forwards to tools/call on the shared session.
    let tools = mcp.tool_definitions().await?;
    let call = |name: &str, args: serde_json::Value| {
        let tool = tools.iter().find(|t| t.name == name).cloned();
        async move {
            let executor = tool.and_then(|t| t.executor).expect("tool is listed");
            executor.invoke(args).await
        }
    };

    // The server sends sampling/createMessage back to us mid-call; the
    // handler answers it with the sampler's completion.
    let sampled = call("trigger-sampling-request", json!({ "prompt": "What colour is the sky?", "maxTokens": 50 })).await?;
    println!("sampling: {sampled}");

    // The server asks for roots/list and reports what we answered.
    let roots = call("get-roots-list", json!({})).await?;
    println!("roots: {roots}");

    mcp.close().await?;
    Ok(())
}
```

`ping` from the server is always answered. Any other server request, including sampling or roots when no handler is set and `elicitation/create`, gets a JSON-RPC "method not found" error rather than silence. The root list is static, so `listChanged` is advertised as `false`; Python's `MCPTool` does not wire roots up at all, so this is one place where the port goes further than the Python package.

## The low-level client

`McpClient::new(Arc<dyn McpTransport>)` is what the three tool types use internally. Use it directly to see raw results, to manage the session yourself, or to test against a fake transport.

| Method | Purpose |
| --- | --- |
| `initialize(name, version)` | Handshake; idempotent, returns `InitializeResult` (`protocol_version`, `capabilities`, `server_info`) |
| `ping()` | Check the server is responsive |
| `list_tools()`, `list_tools_cached()` | Live or cached `Vec<ToolDescriptor>` (`name`, `description`, `input_schema`) |
| `call_tool(name, args)`, `call_tool_value(name, args)` | Raw `CallToolResult`, or the mapped value |
| `list_prompts()`, `list_prompts_cached()`, `get_prompt(name, args)` | Prompts |
| `supports_prompts()`, `server_info()`, `protocol_version()`, `is_initialized()` | State after `initialize` |
| `sampling_handler(h)`, `roots(vec)` | As on the tool types; set before `initialize` |
| `close()` | Kill the stdio child, or DELETE the HTTP session |

Every method except `initialize` fails with "MCP client is not initialized" until the handshake has run. The transports are public too: `McpStdioTransport::spawn(command, &args, &StdioEnv, cwd)` (with `StdioEnv::new().var(...).inherit_var(...)`), `McpStreamableHttpTransport::new(url, headers, timeout)` with `header_map` and `session_id()`, and `McpWebsocketTransport::connect(url, &headers)`. Each takes `with_request_timeout`.

Implementing `McpTransport` (`call`, `notify`, `close`, and optionally `set_server_request_handler` and `set_notification_handler`) gives a fake server for unit tests. The client installs its handlers at construction, so a fake that keeps the notification handler can trigger cache invalidation:

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agent_framework::mcp::{McpClient, McpTransport};
use agent_framework::prelude::*;
use agent_framework::tools::BoxFuture;
use async_trait::async_trait;
use serde_json::{json, Value};

type NotificationHandler = Arc<dyn Fn(String, Value) -> BoxFuture<()> + Send + Sync>;

/// An in-memory MCP "server" for unit tests: canned JSON-RPC results, a
/// counter of tools/list calls, and a way to push a notification.
#[derive(Default)]
struct FakeServer {
    list_calls: AtomicUsize,
    notify: Mutex<Option<NotificationHandler>>,
}

impl FakeServer {
    async fn push(&self, method: &str) {
        let handler = self.notify.lock().unwrap().clone();
        if let Some(handler) = handler {
            handler(method.to_string(), json!({})).await;
        }
    }
}

#[async_trait]
impl McpTransport for FakeServer {
    async fn call(&self, method: &str, params: Value) -> Result<Value> {
        Ok(match method {
            "initialize" => json!({
                "protocolVersion": "2025-06-18",
                "capabilities": { "tools": { "listChanged": true } },
                "serverInfo": { "name": "fake", "version": "0.0.1" }
            }),
            "tools/list" => {
                self.list_calls.fetch_add(1, Ordering::SeqCst);
                json!({ "tools": [
                    { "name": "lookup/order", "description": "Find an order.",
                      "inputSchema": { "type": "object", "properties": { "id": { "type": "string" } } } }
                ] })
            }
            "tools/call" if params["arguments"]["id"] == "404" => json!({
                "content": [{ "type": "text", "text": "no such order" }], "isError": true
            }),
            "tools/call" => json!({ "content": [{ "type": "text", "text": "{\"status\": \"shipped\"}" }] }),
            other => return Err(Error::service(format!("unexpected {other}"))),
        })
    }
    async fn notify(&self, _method: &str, _params: Value) -> Result<()> {
        Ok(())
    }
    async fn close(&self) -> Result<()> {
        Ok(())
    }
    fn set_notification_handler(&self, handler: NotificationHandler) {
        *self.notify.lock().unwrap() = Some(handler);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let server = Arc::new(FakeServer::default());
    let client = McpClient::new(server.clone());
    client.initialize("tests", "0.1.0").await?;

    // The cached listing is what a ToolSource serves on every run.
    let tools = client.list_tools_cached().await?;
    client.list_tools_cached().await?;
    println!("tool: {} | tools/list calls: {}", tools[0].name, server.list_calls.load(Ordering::SeqCst));

    // A list_changed notification empties the cache; the next call refetches.
    server.push("notifications/tools/list_changed").await;
    client.list_tools_cached().await?;
    println!("after list_changed: {} calls", server.list_calls.load(Ordering::SeqCst));

    // A JSON text result is parsed; isError becomes Error::Tool.
    println!("{}", client.call_tool_value("lookup/order", json!({ "id": "42" })).await?);
    match client.call_tool_value("lookup/order", json!({ "id": "404" })).await {
        Err(Error::Tool(message)) => println!("tool error: {message}"),
        other => println!("unexpected: {other:?}"),
    }
    Ok(())
}
```

The notification handler's type is not exported under a name (`BoxedNotificationHandler` lives in a private module), so a transport outside the crate spells it out, as above.

## Lifecycle, errors and reconnecting

A tool type owns one lazily created session. Behaviour worth knowing:

- **Spawn failures** surface on first use as `Error::Service("failed to start MCP server '...': ...")`.
- **A server that dies** fails every request still waiting with "MCP server closed the connection before responding" (stdio), or with the HTTP error (HTTP). There is no automatic reconnection.
- **`close()` is final.** It kills the stdio child (or deletes the HTTP session) but does not reset the tool, so later calls reuse the dead session and fail. To reconnect, build a new tool and a new agent (or a new tool source). With launchers such as `npx`, `close()` kills the launcher, and the real server, its child, exits when its stdin closes, which happens when the tool is dropped.
- **Timeouts** are per request. A timed-out request is removed from the pending table, so a late reply is discarded rather than delivered to a later call.
- **Drop** kills a stdio child (`kill_on_drop`), so a process does not outlive its tool.

In a long-running service, keep the tool behind something you can replace: hold the agent in a `RwLock<Arc<Agent>>`, and on a connection error from `run`, build a fresh `McpStdioTool` and agent and swap them in. Because the tool list is resolved per run, a tool source picks the new server's tools up without further work.

## Hosted MCP

`agent_framework::tools::hosted_mcp(name, url, allowed_tools)` is a different feature: it declares an MCP server that the model provider calls on its side (OpenAI Responses, Anthropic's MCP connector), with its own approval setting via `ToolDefinition::mcp_approval_mode`. No MCP client runs in your process and nothing on this page applies to it. It is covered with the other hosted tools on the [tools page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/).

## Exposing an agent as an MCP server

The crate has no MCP server: there is no counterpart of Python's `agent.as_mcp_server()` or the alpha `agent-framework-hosting-mcp` package. For a client that only needs tools, the streamable HTTP protocol is small enough to serve by hand. This axum handler answers `initialize`, `ping`, `tools/list` and `tools/call`, exposing one agent as one tool, and the crate's own HTTP client calls it:

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

/// A minimal streamable-HTTP MCP server exposing one agent as one tool.
/// The crate has no MCP server, so this speaks the four methods a client needs.
async fn mcp_endpoint(State(agent): State<Arc<Agent>>, Json(request): Json<Value>) -> Json<Value> {
    // Notifications (no id) get an empty 2xx body.
    let Some(id) = request.get("id").cloned() else {
        return Json(json!({}));
    };
    let result = match request["method"].as_str().unwrap_or_default() {
        "initialize" => json!({
            "protocolVersion": "2025-06-18",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "agent-server", "version": "1.0.0" }
        }),
        "ping" => json!({}),
        "tools/list" => json!({ "tools": [{
            "name": "ask_researcher",
            "description": "Ask the research agent a question.",
            "inputSchema": {
                "type": "object",
                "properties": { "question": { "type": "string" } },
                "required": ["question"]
            }
        }] }),
        "tools/call" => {
            let question = request["params"]["arguments"]["question"].as_str().unwrap_or_default();
            match agent.run_once(question).await {
                Ok(reply) => json!({ "content": [{ "type": "text", "text": reply.text() }] }),
                Err(e) => json!({ "content": [{ "type": "text", "text": e.to_string() }], "isError": true }),
            }
        }
        other => {
            return Json(json!({ "jsonrpc": "2.0", "id": id,
                "error": { "code": -32601, "message": format!("method not found: {other}") } }))
        }
    };
    Json(json!({ "jsonrpc": "2.0", "id": id, "result": result }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let researcher = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .name("researcher")
        .instructions("Answer research questions in two sentences.")
        .build();
    let app = Router::new()
        .route("/mcp", post(mcp_endpoint))
        .with_state(Arc::new(researcher));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Any MCP client can now use the agent; here, the crate's own.
    let remote = McpStreamableHttpTool::new("researcher", &url);
    let tools = remote.tool_definitions().await?;
    let reply = tools[0]
        .executor
        .as_ref()
        .unwrap()
        .invoke(json!({ "question": "Why is the sky blue?" }))
        .await?;
    println!("{reply}");
    Ok(())
}
```

This is enough for clients that send each request as a POST and accept a JSON reply. It does not issue session ids, stream, send notifications or support prompts and resources; anything beyond tools needs a full MCP server implementation. For agent-to-agent calls with task semantics, [A2A](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_hosting/#serving-a2a) is the protocol the hosting crate does serve.

## Compared with Python and .NET

| Topic | Rust | Python |
| --- | --- | --- |
| Transports | stdio, streamable HTTP, WebSocket | stdio, streamable HTTP (WebSocket removed) |
| Per-request headers | Fixed at build time | `header_provider` callback |
| Bring your own client | Implement `McpTransport` | Pass your own `mcp` client |
| Roots | Static list, answered | Not wired up |
| Reconnect | Build a new tool | `reset()` and reconnect |
| Serving an agent over MCP | Not provided | `as_mcp_server()` |

Not implemented: the standalone GET stream of the HTTP transport, automatic reconnection, elicitation, progress and logging notifications (logged and ignored), resources and resource templates, completions, the long-running task extension and MCP 2.x negotiation.

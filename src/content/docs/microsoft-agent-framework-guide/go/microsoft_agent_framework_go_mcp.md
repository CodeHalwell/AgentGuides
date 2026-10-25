---
title: "Microsoft Agent Framework (Go) — MCP"
description: "Model Context Protocol in the Go SDK: connecting to MCP servers over stdio, streamable HTTP, SSE and in-memory transports with mcptool, authentication headers, tool naming, filtering and approval, how results are mapped, prompts and resources through the MCP Go SDK, lifecycle, exposing agents and tools as an MCP server, and hosted MCP on the Responses API."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "MCP"
  order: 5
---

The Go SDK's MCP support is a thin layer, `tool/mcptool`, over the official MCP Go SDK (`github.com/modelcontextprotocol/go-sdk/mcp`). It does three things:

- `mcptool.Connect` opens a client session to an MCP server over any `mcp.Transport`.
- `mcptool.ListTools` turns that server's tools into `[]tool.Tool`, which you give to an agent like any other tools.
- `mcptool.AddTool` registers a `tool.FuncTool` (a Go function, or a whole agent via `agenttool`) on an `*mcp.Server`, so MCP clients can call it.

Everything else (transports, prompts, resources, authentication, the server itself) is the MCP Go SDK's API, used directly. There is no MCP-specific agent type and no framework wrapper around prompts or resources. Separately, `hostedtool.MCPServer` asks the OpenAI Responses service to call a remote MCP server itself ([hosted MCP](#hosted-mcp)).

Every program below compiles with `go vet` and `go build` against v%%version=github.com/microsoft/agent-framework-go%% and MCP Go SDK v1.8.0, the version the framework requires. All of them except the hosted-MCP and Microsoft Learn examples were run, with in-process or local MCP servers and a local stand-in for the OpenAI API that calls the first tool it is offered.

## Choosing a transport

| Transport | MCP Go SDK type | Use for |
| --- | --- | --- |
| stdio subprocess | `&mcp.CommandTransport{Command: exec.Command(...)}` | local servers distributed as programs (`npx`, `uvx`, a Go binary) |
| streamable HTTP | `&mcp.StreamableClientTransport{Endpoint: url}` | remote servers; the current HTTP transport |
| SSE (2024-11-05 spec) | `&mcp.SSEClientTransport{Endpoint: url}` | older servers that only offer SSE |
| in memory | `mcp.NewInMemoryTransports()` | a server in the same process, and tests |
| any `io.ReadCloser`/`io.WriteCloser` pair | `&mcp.IOTransport{Reader, Writer}` | custom pipes |

`mcptool.Connect(ctx, transport)` creates an `mcp.Client` named `agent-framework-go-mcp-client` with default options and connects it. If you need client options (a logger, an elicitation or sampling handler, roots), create the client yourself with `mcp.NewClient(impl, opts)` and `client.Connect(ctx, transport, nil)`; `mcptool.ListTools` accepts any `*mcp.ClientSession`.

## Connecting to a server

This program starts an MCP server in-process, connects to it over the in-memory transport, and gives its tools to an agent. The server half is ordinary MCP Go SDK code; swap the transport for a `CommandTransport` or `StreamableClientTransport` and the client half is unchanged.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool/mcptool"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go/v3"
)

type CityIn struct {
	City string `json:"city" jsonschema:"City name"`
}

type TimeOut struct {
	City string `json:"city"`
	Time string `json:"time"`
}

// newServer builds a small MCP server with the MCP Go SDK.
func newServer() *mcp.Server {
	srv := mcp.NewServer(&mcp.Implementation{Name: "clock", Version: "1.0.0"}, nil)
	mcp.AddTool(srv, &mcp.Tool{Name: "local_time", Description: "Current local time in a city."},
		func(ctx context.Context, req *mcp.CallToolRequest, in CityIn) (*mcp.CallToolResult, TimeOut, error) {
			return nil, TimeOut{City: in.City, Time: "14:05"}, nil
		})
	return srv
}

func main() {
	ctx := context.Background()

	serverSide, clientSide := mcp.NewInMemoryTransports()
	serverSession, err := newServer().Connect(ctx, serverSide, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = serverSession.Close() }()

	session, err := mcptool.Connect(ctx, clientSide)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = session.Close() }()

	tools, err := mcptool.ListTools(ctx, session)
	if err != nil {
		log.Fatal(err)
	}
	for _, t := range tools {
		fmt.Printf("tool %s: %s\n", t.Name(), t.Description())
	}

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "Use the clock tools to answer.",
		Config:       agent.Config{Tools: tools},
	})
	resp, err := a.RunText(ctx, "What time is it in Paris?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

`ListTools` pages through the whole tool list and snapshots it: a tool the server adds later is not seen until you call it again. Each returned tool is a `tool.FuncTool` whose `Schema()` is the server's input schema and whose `Call` sends `tools/call` over the session. Calls are made with the run's context.

### A stdio server

`mcp.CommandTransport` runs a program and speaks newline-delimited JSON over its stdin and stdout. The child inherits your environment unless you set `cmd.Env`, so pass only what the server needs, and route its stderr somewhere visible: by default it is discarded, which hides start-up errors.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool/mcptool"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go/v3"
)

type EchoIn struct {
	Text string `json:"text"`
}

// When started with MCP_CHILD=1 this program is the server; otherwise it is the client.
func serve() {
	srv := mcp.NewServer(&mcp.Implementation{Name: "echo", Version: "1.0.0"}, nil)
	mcp.AddTool(srv, &mcp.Tool{Name: "echo", Description: "Echo the text back."},
		func(ctx context.Context, req *mcp.CallToolRequest, in EchoIn) (*mcp.CallToolResult, any, error) {
			return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "echo: " + in.Text}}}, nil, nil
		})
	if err := srv.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
		log.Fatal(err)
	}
}

func main() {
	if os.Getenv("MCP_CHILD") == "1" {
		serve()
		return
	}
	ctx := context.Background()

	// Normally this is something like exec.Command("npx", "-y", "@modelcontextprotocol/server-filesystem", ".").
	cmd := exec.Command(os.Args[0])
	cmd.Env = []string{"MCP_CHILD=1", "PATH=" + os.Getenv("PATH")}
	cmd.Stderr = os.Stderr

	session, err := mcptool.Connect(ctx, &mcp.CommandTransport{Command: cmd})
	if err != nil {
		log.Fatal(err)
	}
	// Close shuts stdin, waits up to TerminateDuration (5s) and then signals the child.
	defer func() { _ = session.Close() }()

	tools, err := mcptool.ListTools(ctx, session)
	if err != nil {
		log.Fatal(err)
	}
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:  "%%model=openai%%",
		Config: agent.Config{Tools: tools},
	})
	resp, err := a.RunText(ctx, "Echo 'hello'.").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

### Streamable HTTP and authentication

`StreamableClientTransport` takes an `HTTPClient`, which is where authentication goes: wrap the transport in an `http.RoundTripper` that adds a header. The transport also has `MaxRetries` (reconnect attempts, default 5, negative to disable), `DisableStandaloneSSE` (skip the long-lived GET stream for server-initiated messages, for servers that mishandle it) and `OAuthHandler` for servers that use the MCP authorisation flow.

The program serves an MCP server with `mcp.NewStreamableHTTPHandler`, protected by a bearer token, and connects to it:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool/mcptool"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go/v3"
)

// bearer adds an Authorization header to every request the MCP client makes.
type bearer struct {
	token string
	next  http.RoundTripper
}

func (b bearer) RoundTrip(r *http.Request) (*http.Response, error) {
	r = r.Clone(r.Context())
	r.Header.Set("Authorization", "Bearer "+b.token)
	return b.next.RoundTrip(r)
}

type SKUIn struct {
	SKU string `json:"sku"`
}

func main() {
	ctx := context.Background()

	srv := mcp.NewServer(&mcp.Implementation{Name: "inventory", Version: "1.0.0"}, nil)
	mcp.AddTool(srv, &mcp.Tool{Name: "stock_level", Description: "Units in stock for a SKU."},
		func(ctx context.Context, req *mcp.CallToolRequest, in SKUIn) (*mcp.CallToolResult, map[string]int, error) {
			return nil, map[string]int{in.SKU: 12}, nil
		})
	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return srv }, nil)
	protected := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer s3cret" {
			http.Error(w, "unauthorised", http.StatusUnauthorized)
			return
		}
		handler.ServeHTTP(w, r)
	})
	ts := httptest.NewServer(protected)
	defer ts.Close()

	session, err := mcptool.Connect(ctx, &mcp.StreamableClientTransport{
		Endpoint:   ts.URL,
		HTTPClient: &http.Client{Transport: bearer{token: "s3cret", next: http.DefaultTransport}},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = session.Close() }()

	tools, err := mcptool.ListTools(ctx, session)
	if err != nil {
		log.Fatal(err)
	}
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:  "%%model=openai%%",
		Config: agent.Config{Tools: tools},
	})
	resp, err := a.RunText(ctx, "How many T-100 do we have?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

A public server needs nothing more than the endpoint. The upstream samples use the Microsoft Learn documentation server:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool/mcptool"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go/v3"
)

func main() {
	ctx := context.Background()
	session, err := mcptool.Connect(ctx, &mcp.StreamableClientTransport{Endpoint: "https://learn.microsoft.com/api/mcp"})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = session.Close() }()

	tools, err := mcptool.ListTools(ctx, session)
	if err != nil {
		log.Fatal(err)
	}
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "Answer from Microsoft Learn and name the page you used.",
		Config:       agent.Config{Name: "DocsAgent", Tools: tools},
	})
	resp, err := a.RunText(ctx, "How do I create a storage account with the az CLI?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

## Names, filtering and approval

MCP tool names may contain characters that model APIs reject. `ListTools` replaces every character outside `[A-Za-z0-9_.-]` with `-`, so `files/read text` is offered to the model as `files-read-text`; the original name is still used when calling the server. If two remote names normalise to the same name, `ListTools` fails with `normalized MCP tool name collision` rather than silently dropping one. Nothing is prefixed, so two servers that both offer `search` collide in the agent: rename one with a small wrapper, as below.

`ListTools` returns everything the server offers. To expose a subset, filter the slice. To require a person's approval, wrap a tool with `tool.ApprovalRequiredFunc`; MCP tools implement `tool.FuncTool`, so a type assertion is all that is needed. The approval round trip is then exactly as for local tools ([tools: approvals](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_tools/#approvals)).

```go
package main

import (
	"context"
	"fmt"
	"log"
	"slices"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/mcptool"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go/v3"
)

type PathIn struct {
	Path string `json:"path"`
}

func fileServer() *mcp.Server {
	srv := mcp.NewServer(&mcp.Implementation{Name: "files", Version: "1.0.0"}, nil)
	for _, name := range []string{"files/read", "files/write", "files/delete"} {
		mcp.AddTool(srv, &mcp.Tool{Name: name, Description: name + " a file"},
			func(ctx context.Context, req *mcp.CallToolRequest, in PathIn) (*mcp.CallToolResult, any, error) {
				return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: req.Params.Name + " ok: " + in.Path}}}, nil, nil
			})
	}
	return srv
}

// renamed changes the name the model sees; calls still go to the original tool.
type renamed struct {
	tool.FuncTool
	name string
}

func (r renamed) Name() string { return r.name }

func main() {
	ctx := context.Background()
	st, ct := mcp.NewInMemoryTransports()
	ss, err := fileServer().Connect(ctx, st, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = ss.Close() }()
	session, err := mcptool.Connect(ctx, ct)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = session.Close() }()

	all, err := mcptool.ListTools(ctx, session)
	if err != nil {
		log.Fatal(err)
	}

	var tools []tool.Tool
	for _, t := range all {
		ft, ok := t.(tool.FuncTool)
		if !ok {
			continue
		}
		switch t.Name() { // normalised names: files-read, files-write, files-delete
		case "files-read":
			tools = append(tools, renamed{ft, "fs_read"})
		case "files-write":
			tools = append(tools, tool.ApprovalRequiredFunc(renamed{ft, "fs_write"}))
		}
		// files-delete is not offered to the model at all.
	}
	fmt.Println("offered:", slices.Collect(func(yield func(string) bool) {
		for _, t := range tools {
			if !yield(t.Name()) {
				return
			}
		}
	}))

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:  "%%model=openai%%",
		Config: agent.Config{Tools: tools},
	})
	s, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := a.RunText(ctx, "Read notes.txt", agent.WithSession(s)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	for c := range resp.Contents() {
		if _, ok := c.(*message.ToolApprovalRequestContent); ok {
			fmt.Println("approval requested")
		}
	}
	fmt.Println(resp)
}
```

## Tool results

`Call` converts the server's `CallToolResult` into framework content, which the tool loop sends back to the model:

| MCP result | Framework value |
| --- | --- |
| `TextContent` | `*message.TextContent` |
| `ImageContent`, `AudioContent` | `*message.DataContent` with the MIME type (default `image/*`, `audio/*`) |
| `ResourceLink` | `*message.URIContent` |
| `EmbeddedResource` with text | `*message.TextContent` |
| `EmbeddedResource` with a blob | `*message.DataContent` named after the resource URI |
| any result with `IsError`, `StructuredContent` or user-defined `_meta` | one `*message.TextContent` holding the whole result as JSON |
| an empty result | `nil` (reported to the model as `Success: Function completed.`) |

Two consequences are worth knowing. A tool that fails on the server (`IsError: true`) is **not** a Go error: the model receives the JSON envelope, including `"isError": true` and the error text, and the loop's consecutive-error counter does not move. Only transport and protocol failures (the server is gone, the session is closed, the context is cancelled) come back as errors, wrapped as `MCP tool call failed: ...`. And a server that returns structured content has the whole envelope sent as text, so the model sees `structuredContent` and `content` side by side; that is normally fine, but it costs tokens.

Each content part's `AdditionalProperties` carries the MCP `_meta`, and `RawRepresentation` the original MCP value, for middleware that wants them.

## Prompts and resources

`mcptool` exposes tools only. Prompts, resources, completions and subscriptions are methods on the MCP Go SDK's `*mcp.ClientSession`, and you decide how their results reach the agent. A common pattern is to read a resource or render a prompt before the run and pass it as instructions or as part of the input:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool/mcptool"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go/v3"
)

func docsServer() *mcp.Server {
	srv := mcp.NewServer(&mcp.Implementation{Name: "style", Version: "1.0.0"}, nil)
	srv.AddResource(&mcp.Resource{URI: "style://guide", Name: "guide", MIMEType: "text/plain"},
		func(ctx context.Context, req *mcp.ReadResourceRequest) (*mcp.ReadResourceResult, error) {
			return &mcp.ReadResourceResult{Contents: []*mcp.ResourceContents{{
				URI: req.Params.URI, MIMEType: "text/plain", Text: "Use British spelling. Prefer short sentences.",
			}}}, nil
		})
	srv.AddPrompt(&mcp.Prompt{Name: "review", Arguments: []*mcp.PromptArgument{{Name: "topic", Required: true}}},
		func(ctx context.Context, req *mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
			return &mcp.GetPromptResult{Messages: []*mcp.PromptMessage{{
				Role:    "user",
				Content: &mcp.TextContent{Text: "Review this text about " + req.Params.Arguments["topic"] + " for clarity."},
			}}}, nil
		})
	return srv
}

func main() {
	ctx := context.Background()
	st, ct := mcp.NewInMemoryTransports()
	ss, err := docsServer().Connect(ctx, st, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = ss.Close() }()
	session, err := mcptool.Connect(ctx, ct)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = session.Close() }()

	// List and read resources.
	for r, err := range session.Resources(ctx, nil) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("resource:", r.URI, r.Name)
	}
	guide, err := session.ReadResource(ctx, &mcp.ReadResourceParams{URI: "style://guide"})
	if err != nil {
		log.Fatal(err)
	}

	// Render a prompt.
	prompt, err := session.GetPrompt(ctx, &mcp.GetPromptParams{
		Name: "review", Arguments: map[string]string{"topic": "goroutines"},
	})
	if err != nil {
		log.Fatal(err)
	}
	var promptText strings.Builder
	for _, m := range prompt.Messages {
		if t, ok := m.Content.(*mcp.TextContent); ok {
			promptText.WriteString(t.Text)
		}
	}

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "You are an editor. House style: " + guide.Contents[0].Text,
	})
	resp, err := a.RunText(ctx, promptText.String()+"\n\nText: Goroutines is cheap threads.").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

To give the model on-demand access to resources instead, write a small function tool that calls `session.ReadResource`. Sampling (`ClientOptions.CreateMessageHandler`) and roots are supported by the MCP Go SDK if you build the client yourself, but both are deprecated in the current MCP specification, and the framework does not connect them to an agent.

## Lifecycle, errors and reconnecting

An `*mcp.ClientSession` is a live connection. Keep it open for as long as an agent may call its tools, and close it afterwards; for a `CommandTransport`, `Close` also stops the child process (stdin is closed, then the process gets `TerminateDuration`, default five seconds, before it is signalled). `session.Wait()` blocks until the server ends the session.

The tools returned by `ListTools` hold the session. If the server goes away, every later call fails with `MCP tool call failed: ...`, the loop reports `Error: Function failed.` to the model, and after the consecutive-error limit the run fails. There is no automatic reconnect at the framework level (the streamable HTTP transport retries a dropped stream up to `MaxRetries` times on its own). For long-running services, reconnect and rebuild the tool list when a call fails with a closed-session error, and pass the new tools per run with `agent.WithTool` so the agent need not be rebuilt:

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/mcptool"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go/v3"
)

type PingIn struct {
	Host string `json:"host"`
}

func newServer() *mcp.Server {
	srv := mcp.NewServer(&mcp.Implementation{Name: "net", Version: "1.0.0"}, nil)
	mcp.AddTool(srv, &mcp.Tool{Name: "ping", Description: "Ping a host."},
		func(ctx context.Context, req *mcp.CallToolRequest, in PingIn) (*mcp.CallToolResult, any, error) {
			return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: in.Host + " is up"}}}, nil, nil
		})
	return srv
}

// mcpSource keeps one session and rebuilds it on demand.
type mcpSource struct {
	mu      sync.Mutex
	connect func(context.Context) (*mcp.ClientSession, error)
	session *mcp.ClientSession
	tools   []tool.Tool
}

func (s *mcpSource) Tools(ctx context.Context) ([]tool.Tool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.session != nil {
		if err := s.session.Ping(ctx, nil); err == nil {
			return s.tools, nil
		}
		_ = s.session.Close()
		s.session = nil
	}
	session, err := s.connect(ctx)
	if err != nil {
		return nil, err
	}
	tools, err := mcptool.ListTools(ctx, session)
	if err != nil {
		_ = session.Close()
		return nil, err
	}
	s.session, s.tools = session, tools
	return tools, nil
}

func (s *mcpSource) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.session == nil {
		return nil
	}
	return s.session.Close()
}

func main() {
	ctx := context.Background()
	var serverSessions []*mcp.ServerSession
	src := &mcpSource{connect: func(ctx context.Context) (*mcp.ClientSession, error) {
		st, ct := mcp.NewInMemoryTransports()
		ss, err := newServer().Connect(ctx, st, nil)
		if err != nil {
			return nil, err
		}
		serverSessions = append(serverSessions, ss)
		return mcptool.Connect(ctx, ct)
	}}
	defer func() { _ = src.Close() }()

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{Model: "%%model=openai%%"})

	ask := func(q string) {
		tools, err := src.Tools(ctx)
		if err != nil {
			log.Fatal(err)
		}
		opts := make([]agent.Option, 0, len(tools))
		for _, t := range tools {
			opts = append(opts, agent.WithTool(t))
		}
		resp, err := a.RunText(ctx, q, opts...).Collect()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(resp)
	}

	ask("Is example.com up?")
	_ = serverSessions[0].Close() // the server goes away
	ask("Is example.org up?")     // Ping fails, so the source reconnects first
	fmt.Println("connections made:", len(serverSessions))
	if len(serverSessions) != 2 {
		log.Fatal(errors.New("expected a reconnect"))
	}
}
```

## Exposing tools and agents as an MCP server

`mcptool.AddTool(server, funcTool)` registers any `tool.FuncTool` on an `*mcp.Server`: its name, description, input schema and output schema become the MCP tool definition. Combined with `agenttool.New`, that publishes a whole agent as one MCP tool with a single `query` argument, which any MCP client (an IDE, Claude Desktop, another agent) can call.

How results are returned:

| The tool returns | MCP result |
| --- | --- |
| an error | `IsError: true` with the error text; not a protocol error |
| a struct or map | text content holding the JSON, plus `structuredContent` |
| an array, when the output schema is an array | text content holding the JSON array, and `structuredContent: {"result": [...]}`, because MCP structured content must be an object; the advertised output schema is wrapped the same way |
| `string` | text content |
| `mcp.Content`, `[]mcp.Content`, `*mcp.CallToolResult` | passed through as they are |
| `message.Content` values | converted back to MCP content (text, images, resources) |

```go
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool/agenttool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/microsoft/agent-framework-go/tool/mcptool"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go/v3"
)

type ConvertIn struct {
	Celsius float64 `json:"celsius"`
}

type ConvertOut struct {
	Fahrenheit float64 `json:"fahrenheit"`
}

var convert = functool.MustNew(functool.Config{Name: "c_to_f", Description: "Convert Celsius to Fahrenheit."},
	func(ctx context.Context, in ConvertIn) (ConvertOut, error) {
		return ConvertOut{Fahrenheit: in.Celsius*9/5 + 32}, nil
	})

func main() {
	ctx := context.Background()

	reviewer := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "You review Go code for correctness and idiom. Be brief.",
		Config: agent.Config{
			Name:        "go_reviewer", // the MCP tool name
			Description: "Reviews a Go snippet and lists problems.",
		},
	})

	srv := mcp.NewServer(&mcp.Implementation{Name: "team-tools", Version: "1.0.0"}, nil)
	mcptool.AddTool(srv, agenttool.New(reviewer, agenttool.Config{}))
	mcptool.AddTool(srv, convert)

	// Serve over streamable HTTP. For a local server that an IDE launches, use
	// srv.Run(ctx, &mcp.StdioTransport{}) instead.
	ts := httptest.NewServer(mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return srv }, nil))
	defer ts.Close()

	// Any MCP client can now call the tools. Here, the MCP Go SDK's own client.
	client := mcp.NewClient(&mcp.Implementation{Name: "check", Version: "1.0.0"}, nil)
	cs, err := client.Connect(ctx, &mcp.StreamableClientTransport{Endpoint: ts.URL}, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = cs.Close() }()

	for t, err := range cs.Tools(ctx, nil) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("served:", t.Name)
	}
	res, err := cs.CallTool(ctx, &mcp.CallToolParams{Name: "c_to_f", Arguments: map[string]any{"celsius": 21}})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("c_to_f: %s (structured: %v)\n", res.Content[0].(*mcp.TextContent).Text, res.StructuredContent)

	res, err = cs.CallTool(ctx, &mcp.CallToolParams{Name: "go_reviewer", Arguments: map[string]any{"query": "x := []int{}; x[0] = 1"}})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("go_reviewer:", res.Content[0].(*mcp.TextContent).Text)
}
```

The served tool runs the agent with the MCP request's context, so a client that cancels the call cancels the run. Each call is a fresh, sessionless agent run. For a hosted server, `mcp.NewStreamableHTTPHandler` options include `Stateless` (no `Mcp-Session-Id`, suitable behind a load balancer), `JSONResponse`, `SessionTimeout` and `EventStore` for stream resumption; it also rejects requests to localhost listeners that carry a non-localhost `Host` header, as protection against DNS rebinding. [Hosting and A2A](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_hosting_and_a2a/#mcp-over-http) covers serving MCP alongside other endpoints in a production HTTP server.

## Hosted MCP

With `hostedtool.MCPServer` the model service, not your process, connects to a remote MCP server. Only the OpenAI Responses API (and Foundry, which uses it) supports this; other providers skip the marker.

| Field | Sent as |
| --- | --- |
| `ServerName` | `server_label` |
| `ServerAddress` | `server_url`, if it is an `http` or `https` URL |
| `ServerDescription` | `server_description` |
| `AllowedTools` | `allowed_tools` |
| `Headers` | `headers` |
| `Authorization` | `authorization` |
| `ApprovalMode` | `require_approval`: `hostedtool.MCPApprovalAlways` or `MCPApprovalNever`. Unset leaves the service default, which for the Responses API is to require approval for every call |

When the service wants approval, the run ends with a `*message.ToolApprovalRequestContent` whose `ToolCall` is a `*message.MCPServerToolCallContent` (server, tool name and arguments). Answer it with `CreateResponse`, as for local tools; the provider sends it back as an `mcp_approval_response`. Completed calls appear in the response as `MCPServerToolCallContent` and `MCPServerToolResultContent`.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/hostedtool"
	"github.com/openai/openai-go/v3"
)

func main() {
	ctx := context.Background()
	a := openaiprovider.NewResponsesAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "Answer from Microsoft Learn.",
		Config: agent.Config{Tools: []tool.Tool{&hostedtool.MCPServer{
			ServerName:    "learn",
			ServerAddress: "https://learn.microsoft.com/api/mcp",
			AllowedTools:  []string{"microsoft_docs_search"},
			ApprovalMode:  hostedtool.MCPApprovalAlways,
		}}},
	})

	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := a.RunText(ctx, "How do I enable soft delete on blob storage?", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}

	for {
		var answers []message.Content
		for c := range resp.Contents() {
			if req, ok := c.(*message.ToolApprovalRequestContent); ok {
				if call, ok := req.ToolCall.(*message.MCPServerToolCallContent); ok {
					fmt.Printf("approving %s/%s %s\n", call.ServerName, call.Name, call.Arguments)
				}
				answers = append(answers, req.CreateResponse(true, ""))
			}
		}
		if len(answers) == 0 {
			break
		}
		resp, err = a.RunMessage(ctx, message.New(answers...), agent.WithSession(session)).Collect()
		if err != nil {
			log.Fatal(err)
		}
	}

	for c := range resp.Contents() {
		if call, ok := c.(*message.MCPServerToolCallContent); ok {
			fmt.Println("service called", call.ServerName, call.Name)
		}
	}
	fmt.Println(resp)
}
```

Hosted MCP suits public or internet-reachable servers and keeps the tool traffic off your process. Use `mcptool` instead when the server is local, on a private network, needs credentials you do not want to hand to the model provider, or when you want function middleware and the framework's approval rules to apply to every call.

## Compared with Python and .NET

- **Thin wrapper.** Python's `MCPStdioTool`, `MCPStreamableHTTPTool` and `MCPWebsocketTool`, and .NET's `McpClient` helpers, manage connection lifetime and expose prompts. Go gives you the MCP Go SDK session and converts only tools; prompts and resources are read with the session's own methods.
- **No WebSocket transport** in the MCP Go SDK, so no Go equivalent of `MCPWebsocketTool`.
- **No per-tool approval mode or name prefix on the client.** Filter, rename and wrap tools yourself, as above.
- **Serving** is similar to Python's `agent.as_mcp_server()`: `mcptool.AddTool` plus `agenttool.New` on an `*mcp.Server`, served over stdio or streamable HTTP by the MCP Go SDK.

The [Python MCP page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_mcp/) covers its client classes in more depth.

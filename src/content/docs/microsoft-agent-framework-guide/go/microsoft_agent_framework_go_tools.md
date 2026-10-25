---
title: "Microsoft Agent Framework (Go) — Tools"
description: "Tools in the Go SDK: typed function tools and their JSON schemas, hand-written tools, the automatic tool loop and its limits, tool modes, approvals and standing approval rules, agents as tools, provider-hosted web search, file search, code interpreter and MCP, and the local shell tool."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "Tools"
  order: 4
---

A tool is anything that implements `tool.Tool`, which is two methods: `Name()` and `Description()`. What the framework does with it depends on which further interfaces it implements. A **function tool** (`tool.FuncTool`) has an input schema and a `Call` method, and the framework runs it locally when the model asks. A **hosted tool** (`hostedtool.WebSearch` and friends) has neither, and is a marker telling the provider's service that it may do something itself. Both kinds go in the same `agent.Config.Tools` slice, or are added to one run with `agent.WithTool`.

This page covers function tools, the loop that runs them, approvals, agents as tools, hosted tools and the shell tool. MCP servers as tool sources have [their own page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_mcp/), and function middleware, which wraps every tool call, is on the [middleware page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_middleware/#function-middleware).

Every program compiles with `go vet` and `go build` against v%%version=github.com/microsoft/agent-framework-go%%. The function-tool, loop, approval, agent-as-tool and shell examples were run against a local stand-in for the OpenAI API that calls the first tool it is offered, with arguments generated from the tool's schema, and then answers with the tool's result. The hosted-tool examples were only compiled.

## How tools are represented

| Interface | Methods | Implemented by |
| --- | --- | --- |
| `tool.Tool` | `Name() string`, `Description() string` | every tool, including the hosted markers |
| `tool.SchemaTool` | adds `Schema() any`, `ReturnSchema() any` | tools the model can call with arguments |
| `tool.FuncTool` | adds `Call(ctx, args string) (any, error)` | `functool`, `agenttool`, MCP tools, `shelltool.Local`, your own types |
| `tool.ApprovalRequiredTool` | adds `ApprovalRequired() bool` | `tool.ApprovalRequiredFunc` wrappers, `shelltool.Local`, skill tools |

Providers decide what to send by type-switching on the tools in the run options: a `tool.FuncTool` becomes a function declaration, a hosted marker the provider understands becomes the service's own tool type, and anything else is skipped without an error. `Schema()` returns a JSON-schema value (a `*jsonschema.Schema` from `github.com/google/jsonschema-go`, or a `map[string]any`), which each provider converts to its wire format.

## Function tools

`functool.New` and `functool.MustNew` turn a typed function `func(context.Context, In) (Out, error)` into a `tool.FuncTool`. `MustNew` panics instead of returning an error, which suits package-level variables.

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
)

type ForecastIn struct {
	City  string   `json:"city" jsonschema:"City name, for example Leeds"`
	Days  int      `json:"days,omitempty" jsonschema:"Number of days, 1 to 7"`
	Units string   `json:"units,omitempty" jsonschema:"celsius or fahrenheit"`
	Extra []string `json:"extra,omitempty" jsonschema:"Optional extras such as wind or uv"`
}

type Day struct {
	Summary string  `json:"summary"`
	HighC   float64 `json:"high_c"`
}

type ForecastOut struct {
	City string `json:"city"`
	Days []Day  `json:"days"`
}

var forecast = functool.MustNew(functool.Config{
	Name:        "get_forecast",
	Description: "Get the weather forecast for a city.",
}, func(ctx context.Context, in ForecastIn) (ForecastOut, error) {
	if strings.EqualFold(in.City, "atlantis") {
		return ForecastOut{}, errors.New("no weather station in " + in.City)
	}
	days := max(in.Days, 1)
	out := ForecastOut{City: in.City}
	for range days {
		out.Days = append(out.Days, Day{Summary: "showers", HighC: 14})
	}
	return out, nil
})

func main() {
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "Use the tools to answer weather questions.",
		Config:       agent.Config{Tools: []tool.Tool{forecast}},
	})

	resp, err := a.RunText(context.Background(), "What's the weather in Paris this week?").Collect()
	if err != nil {
		log.Fatal(err)
	}

	// The response holds the whole exchange: the call, the result, then the answer.
	for _, m := range resp.Messages {
		for _, c := range m.Contents {
			switch c := c.(type) {
			case *message.FunctionCallContent:
				fmt.Printf("call   %s %s\n", c.Name, c.Arguments)
			case *message.FunctionResultContent:
				fmt.Printf("result %v\n", c.Result)
			}
		}
	}
	fmt.Println("answer", resp)
}
```

The schema is derived from the input and output types with `github.com/google/jsonschema-go`. For `ForecastIn` the model is sent:

```json
{
  "type": "object",
  "properties": {
    "city":  { "type": "string", "description": "City name, for example Leeds" },
    "days":  { "type": "integer", "description": "Number of days, 1 to 7" },
    "units": { "type": "string", "description": "celsius or fahrenheit" },
    "extra": { "type": ["null", "array"], "items": { "type": "string" },
               "description": "Optional extras such as wind or uv" }
  },
  "required": ["city"],
  "additionalProperties": false
}
```

The rules that follow from this:

- **Property names** come from the `json` tag, and the `jsonschema` tag is the property's **description**, nothing more. There is no tag syntax for enums, ranges or patterns; put constraints in the description, or write the schema yourself (below).
- **Required** means "no `omitempty`". A field without `omitempty` is required, even a pointer, which becomes `["null", <type>]` and so must be sent but may be `null`. Slices and maps are also nullable.
- **`additionalProperties: false`** is set on every struct, so an argument the model invents is an error.
- **Arguments are validated** against the schema before your function runs. A missing required field or an extra property fails with an error such as `validation error: validating root: required: missing properties: ["city"]`, and your function is never called.
- **Non-struct inputs** are wrapped. `func(ctx, string)` gets the schema `{"properties": {"Arg0": {"type": "string"}}, "required": ["Arg0"]}`, so the model must send `{"Arg0": "..."}`. A struct with a descriptive field name is nearly always better. `In` of type `any` is rejected when the tool is built.
- **The output** is checked against the output schema and has defaults applied before it is returned; a typed nil pointer becomes the zero value. Gemini sends the output schema to the model as `responseJsonSchema`; the other providers do not use it.
- **The result** sent back to the model is the returned value encoded as JSON (a string stays a string). A tool that returns `nil` with no error is reported as `Success: Function completed.`.
- **The `context.Context`** is the run's context, so deadlines and cancellation reach your code.

### What the model sees when a tool fails

When a tool returns an error, the loop does not fail the run. It records a `FunctionResultContent` whose `Result` is the text `Error: Function failed.` (or `Error: Function failed. Exception: <your error>` with `IncludeDetailedErrors`) and whose `Error` field holds your error, and lets the model carry on. Validation failures count as errors in the same way.

**The model sees your error text anyway.** The `toolautocall` documentation says the generic result keeps details from the model, but in v%%version=github.com/microsoft/agent-framework-go%% every built-in provider serialises the `Error` field, not the `Result`, whenever `Error` is set. So `IncludeDetailedErrors` makes no visible difference, and the raw error reaches the model. Checked in the source and against the local stand-in with a tool returning `pq: password authentication failed for user admin`:

| Provider | What the model receives |
| --- | --- |
| OpenAI Responses, Foundry | `function_call_output`: `Error: pq: password authentication failed for user admin` |
| OpenAI Chat Completions | tool message: `pq: password authentication failed for user admin` (no prefix) |
| Anthropic | `tool_result` text: `Error: pq: password authentication failed for user admin` |
| Gemini | `functionResponse`: `{"error": "pq: password authentication failed for user admin"}` |

Errors from databases, HTTP clients and the file system often contain host names, user names or file paths, so do not let them through unfiltered. Use the function middleware on the [production page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_production/#limits-on-the-tool-loop): it logs the real error and returns a bland one, which then is what the model sees. A round counts as failing when any call in it returned an error, and once the number of consecutive failing rounds exceeds `MaximumConsecutiveErrorsPerRequest` (three by default, so on the fourth) the loop gives up and the run returns that round's errors, joined with `errors.Join` when there are several.

If you want the model to see a particular message, return it as the result instead of an error: `return "no weather station in Atlantis; try a nearby city", nil`.

### Implementing `tool.FuncTool` yourself

`functool` needs the input type at compile time. When the schema is only known at run time (loaded from configuration, generated from a database table, proxied from another system), implement the interface directly. `Call` receives the raw JSON arguments and nothing is validated for you:

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"slices"
	"strings"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/openai/openai-go/v3"
)

// lookupTool answers questions about one table whose columns are known only at run time.
type lookupTool struct {
	table   string
	columns []string
	rows    []map[string]string
}

func (t *lookupTool) Name() string        { return "lookup_" + t.table }
func (t *lookupTool) Description() string { return "Find rows in the " + t.table + " table by column value." }

func (t *lookupTool) Schema() any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"column": map[string]any{"type": "string", "enum": t.columns},
			"value":  map[string]any{"type": "string"},
		},
		"required":             []string{"column", "value"},
		"additionalProperties": false,
	}
}

func (t *lookupTool) ReturnSchema() any { return nil }

func (t *lookupTool) Call(ctx context.Context, args string) (any, error) {
	var in struct{ Column, Value string }
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return nil, fmt.Errorf("bad arguments: %w", err)
	}
	if !slices.Contains(t.columns, in.Column) {
		return fmt.Sprintf("unknown column %q; use one of %s", in.Column, strings.Join(t.columns, ", ")), nil
	}
	var hits []map[string]string
	for _, r := range t.rows {
		if strings.EqualFold(r[in.Column], in.Value) {
			hits = append(hits, r)
		}
	}
	return hits, nil
}

func main() {
	products := &lookupTool{
		table:   "products",
		columns: []string{"sku", "name", "colour"},
		rows: []map[string]string{
			{"sku": "T-100", "name": "TrailRunner tent", "colour": "green"},
			{"sku": "B-200", "name": "Summit bag", "colour": "red"},
		},
	}

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:  "%%model=openai%%",
		Config: agent.Config{Tools: []tool.Tool{products}},
	})
	resp, err := a.RunText(context.Background(), "Which products are green?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

`json.Unmarshal` matches field names case-insensitively, which is why the anonymous struct with `Column` and `Value` reads `column` and `value`. `functool.Handler`, a `func(ctx, args string) (any, error)` type, exists in the package and is mentioned by an error message, but no constructor accepts it in v%%version=github.com/microsoft/agent-framework-go%%; a small type like the one above is the way to write a dynamic tool.

## The tool loop

The OpenAI, Foundry, Anthropic and Gemini providers install the `agent/harness/toolautocall` middleware inside the agent. When a model response contains `FunctionCallContent`, it finds the tool by name, calls it, appends a tool-role message with the `FunctionResultContent`, and calls the model again. It repeats until a response has no function calls or a limit is reached. All of this happens inside one run, so the response you collect contains the calls, the results and the final answer, and the session stores them all.

The loop is configured through the provider's `AgentConfig.ToolAutoCall`:

| `toolautocall.Config` field | Default | Effect |
| --- | --- | --- |
| `MaximumIterationsPerRequest *int` | 40 | rounds of tool calls per run. When reached, one last model call is made without any function tools, so the model has to answer in text. `new(0)` switches the loop off: function calls are returned to you unexecuted. Negative is an error |
| `MaximumConsecutiveErrorsPerRequest *int` | 3 | consecutive rounds with at least one failed call that are tolerated; one more fails the run with that round's errors. `new(0)` fails on the first tool error. A round without errors resets the count |
| `IncludeDetailedErrors` | false | append the error text to the generic result. The built-in providers send the raw error regardless ([see above](#what-the-model-sees-when-a-tool-fails)) |
| `AllowConcurrentInvocations` | false | run several calls from one response in parallel goroutines. Your tools must then be safe for concurrent use |
| `TerminateOnUnknownCalls` | false | when the model calls a tool that does not exist, stop and return the response to you, instead of telling the model `Error: Requested function "x" not found.` |
| `AdditionalTools []tool.Tool` | none | tools the loop may execute but does not advertise to the model, for providers configured with tool declarations some other way |
| `Logger`, `LogSensitiveData` | the agent's, when `ToolAutoCall` is nil | invocation logs; arguments and results only with `LogSensitiveData` |
| `NewID func() string` | UUIDs | IDs for messages the loop synthesises |
| `DisableApprovalResponseBinding` | false | see [approvals](#approvals) |
| `DisableApprovalNotRequiredFunctionBypassing` | false | see [approvals](#approvals) |

Setting `ToolAutoCall` replaces the defaults wholesale, so a non-nil config with only `AllowConcurrentInvocations: true` still gets 40 rounds and three errors (nil pointers mean default), but loses the agent's logger unless you set `Logger` too.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"os"
	"sync/atomic"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/harness/toolautocall"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
)

type StockIn struct {
	SKU string `json:"sku"`
}

var calls atomic.Int32

var stock = functool.MustNew(functool.Config{Name: "check_stock", Description: "Units in stock for a SKU."},
	func(ctx context.Context, in StockIn) (int, error) {
		calls.Add(1)
		return 12, nil
	})

var price = functool.MustNew(functool.Config{Name: "get_price", Description: "Price in pounds for a SKU."},
	func(ctx context.Context, in StockIn) (float64, error) { return 89.5, nil })

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model: "%%model=openai%%",
		ToolAutoCall: &toolautocall.Config{
			MaximumIterationsPerRequest:        new(5),
			MaximumConsecutiveErrorsPerRequest: new(1),
			AllowConcurrentInvocations:         true,
			IncludeDetailedErrors:              true,
			Logger:                             logger,
		},
		Config: agent.Config{Tools: []tool.Tool{stock}},
	})

	ctx := context.Background()

	// Force a tool call, and add a tool for this run only.
	resp, err := a.RunText(ctx, "Is T-100 in stock, and what does it cost?",
		agent.WithTool(price),
		agent.WithToolMode(tool.RequireTool("get_price")),
	).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)

	// Forbid tool calls for one run.
	resp, err = a.RunText(ctx, "Say hello.", agent.WithToolMode(tool.ToolModeNone)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp, "| stock calls so far:", calls.Load())

	// With the loop switched off, the call comes back to you unexecuted.
	manual := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		ToolAutoCall: &toolautocall.Config{MaximumIterationsPerRequest: new(0)},
		Config:       agent.Config{Tools: []tool.Tool{stock}},
	})
	resp, err = manual.RunText(ctx, "Is T-100 in stock?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	for c := range resp.Contents() {
		if call, ok := c.(*message.FunctionCallContent); ok {
			fmt.Println("model asked for", call.Name, call.Arguments, "id", call.CallID)
		}
	}
}
```

### Tool modes

`agent.WithToolMode` sets the tool choice for one run, or for every run if put in `RunOptions`:

| Mode | Meaning | OpenAI | Anthropic | Gemini |
| --- | --- | --- | --- | --- |
| `tool.ToolModeAuto` | the model decides | `auto` | `auto` | `AUTO` |
| `tool.ToolModeNone` | no tool calls | `none` | `none` | `NONE` |
| `tool.ToolModeRequired` | at least one tool call | `required` | `any` | `ANY` |
| `tool.RequireTool("name")` | this tool | named function | named tool | `ANY` with `allowedFunctionNames` |

A required mode applies to the first model call only. After the first round of tool calls the loop removes the tool-mode option, so the model is free to answer instead of calling tools for ever. `RequireTool` panics on a blank name.

### Per-run tools

`agent.WithTool(t)` adds a tool for one run, on top of `Config.Tools`. Context providers can also add tools per run, which is how the [memory, retrieval and skills providers](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_sessions_and_memory/) expose their search and load tools. Two tools with the same name confuse both the model and the loop's lookup, so keep names unique; the loop uses the first match.

## Approvals

Wrap a function tool in `tool.ApprovalRequiredFunc` to require a person's decision before it runs. When the model calls it, the loop does not execute it. The run ends with a `*message.ToolApprovalRequestContent` in the response; you answer with `req.CreateResponse(approved, reason)` in a new run on the **same session**:

```go
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
)

type RefundIn struct {
	OrderID string  `json:"order_id"`
	Amount  float64 `json:"amount"`
}

var refund = tool.ApprovalRequiredFunc(functool.MustNew(
	functool.Config{Name: "issue_refund", Description: "Refund an order."},
	func(ctx context.Context, in RefundIn) (string, error) {
		return fmt.Sprintf("refunded %.2f on %s", in.Amount, in.OrderID), nil
	}))

func main() {
	ctx := context.Background()
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "You handle refunds.",
		Config:       agent.Config{Tools: []tool.Tool{refund}},
	})

	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	stdin := bufio.NewReader(os.Stdin)

	resp, err := a.RunText(ctx, "Refund order 1182 in full, it was 40 pounds.", agent.WithSession(session)).Collect()
	for err == nil {
		var answers []message.Content
		for c := range resp.Contents() {
			req, ok := c.(*message.ToolApprovalRequestContent)
			if !ok {
				continue
			}
			call := req.ToolCall.(*message.FunctionCallContent)
			fmt.Printf("Allow %s(%s)? [y/N] ", call.Name, call.Arguments)
			line, _ := stdin.ReadString('\n')
			if strings.EqualFold(strings.TrimSpace(line), "y") {
				answers = append(answers, req.CreateResponse(true, ""))
			} else {
				answers = append(answers, req.CreateResponse(false, "the operator declined"))
			}
		}
		if len(answers) == 0 {
			break // no more approvals pending: resp is the final answer
		}
		resp, err = a.RunMessage(ctx, message.New(answers...), agent.WithSession(session)).Collect()
	}
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Piped an empty line, the program declines; the model is told `Tool call invocation rejected. the operator declined` and answers accordingly. How the round trip behaves:

- **Each request has a `RequestID`**, and the response must answer that ID. `CreateResponse` copies the ID and the tool call for you.
- **Approve or reject every pending request.** The next run must contain a response for each request the previous run returned, in one message or several.
- **Binding.** The loop records the requests it surfaced in the session and checks each response against them, so a client cannot approve a call that was never requested or alter its arguments. That is why approvals need a session. Without one there is nothing to match the response against, and it is dropped without an error: in testing, the follow-up run called the model with an empty message list and the tool never ran. `DisableApprovalResponseBinding` switches the check off, for callers that persist approval state themselves.
- **Safe calls in the same batch.** If the model asks for an approval-required tool and an ordinary tool in one response, the ordinary call is held back and run automatically once the approvals come in, so you only see the request that needs a decision. `DisableApprovalNotRequiredFunctionBypassing` instead surfaces every call in the batch as an approval request.
- **Streaming** works the same way: the `ToolApprovalRequestContent` arrives in an update, and the stream ends.
- **A second approval on a local-history session can fail.** Against the local stand-in, on a session that keeps its history locally (Chat Completions, Anthropic, Gemini, or Responses with `DisableStoreOutput`), the run *after* a completed approval failed with `ToolApprovalRequestContent found with ToolCall.CallID(s) '...' that have no matching ToolApprovalResponseContent`, naming the earlier, already answered request. Sessions with service-held history were not affected. It was seen against a stand-in only, not a real model; [sessions and memory](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_sessions_and_memory/#approvals-and-local-history) has the details and a workaround.
- **The model never sees the request.** Approval content is removed before the next model call and replaced by the function call and its result, or the rejection text.

### Standing approvals with `toolapproval`

For "approve, and don't ask again" behaviour, add the `agent/harness/toolapproval` middleware to the agent's middleware. It sits outside the tool loop, keeps standing rules in the session, and surfaces pending requests one at a time:

| Response | Effect |
| --- | --- |
| `req.CreateResponse(true, "")` / `(false, reason)` | approve or reject this call only |
| `req.AlwaysApproveToolResponse()` | approve, and approve every later call to this tool in the session |
| `req.AlwaysApproveToolWithArgumentsResponse()` | approve, and approve later calls with exactly these arguments |

`toolapproval.Config.AutoApprovalRules` adds rules of your own, evaluated after the standing rules and before asking. A rule receives the call, the agent, the session, the request messages and the options, and returns `(approved, error)`; an error fails the run. `toolapproval.AllToolsAutoApprovalRule` approves everything and is only for trusted tool sets. `MaxAutoApprovalIterations` (default 40) caps how often the middleware re-runs the agent in one run while everything is being auto-approved.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/harness/toolapproval"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
)

type PathIn struct {
	Path string `json:"path"`
}

var readFile = tool.ApprovalRequiredFunc(functool.MustNew(
	functool.Config{Name: "read_file", Description: "Read a file from the workspace."},
	func(ctx context.Context, in PathIn) (string, error) { return "contents of " + in.Path, nil }))

// Files under docs/ never need a person.
func docsAreSafe(ctx context.Context, rc *toolapproval.ToolAutoApprovalRuleContext) (bool, error) {
	return rc.FunctionCall.Name == "read_file" && strings.Contains(rc.FunctionCall.Arguments, `"docs/`), nil
}

func run(ctx context.Context, a *agent.Agent, s *agent.Session, msg *message.Message) *agent.Response {
	resp, err := a.RunMessage(ctx, msg, agent.WithSession(s)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	return resp
}

func pending(resp *agent.Response) *message.ToolApprovalRequestContent {
	for c := range resp.Contents() {
		if req, ok := c.(*message.ToolApprovalRequestContent); ok {
			return req
		}
	}
	return nil
}

func main() {
	ctx := context.Background()
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model: "%%model=openai%%",
		Config: agent.Config{
			Tools: []tool.Tool{readFile},
			Middlewares: []agent.Middleware{toolapproval.New(toolapproval.Config{
				AutoApprovalRules: []toolapproval.AutoApprovalRule{docsAreSafe},
			})},
		},
	})
	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// First call: a person approves, and asks not to be asked again.
	resp := run(ctx, a, session, message.NewText("Read scratch.txt"))
	if req := pending(resp); req != nil {
		fmt.Println("asked once for", req.ToolCall.(*message.FunctionCallContent).Name)
		resp = run(ctx, a, session, message.New(req.AlwaysApproveToolResponse()))
	}
	fmt.Println(resp)

	// Second call to the same tool: approved by the standing rule, no request.
	resp = run(ctx, a, session, message.NewText("Read scratch.txt again"))
	fmt.Println("pending:", pending(resp) != nil, "|", resp)
}
```

Standing rules live in the session's state, so they survive serialising the session and are scoped to it: a new session asks again. The middleware works with tools that require approval; it does not add approval to tools that do not.

## An agent as a tool

`agenttool.New(a, agenttool.Config{})` wraps an `*agent.Agent` as a `tool.FuncTool`. The tool takes one required string argument, `query`, runs the agent with `RunText`, and returns the response text. Its name is the agent's `Name` (or its `ID` if the name is empty) with every run of characters outside `[0-9A-Za-z]` replaced by `_`, so "Billing Agent" becomes `Billing_Agent`. Its description is the agent's `Description`, or "Invoke an agent to retrieve some information." when that is empty, so give specialists a good description: it is all the manager has to choose by.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/agenttool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
)

type OrderIn struct {
	OrderID string `json:"order_id"`
}

var orderStatus = functool.MustNew(functool.Config{Name: "order_status", Description: "Status of an order."},
	func(ctx context.Context, in OrderIn) (string, error) { return "order " + in.OrderID + " shipped", nil })

func main() {
	client := openai.NewClient()

	// A specialist with its own tools and instructions.
	orders := openaiprovider.NewAgent(client, openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "You answer questions about orders. Use order_status.",
		Config: agent.Config{
			Name:        "Order Desk",
			Description: "Answers questions about the status of customer orders.",
			Tools:       []tool.Tool{orderStatus},
		},
	})

	// The manager sees the specialist as a tool called Order_Desk.
	ordersTool := agenttool.New(orders, agenttool.Config{
		RunOptions: []agent.Option{agent.WithInstructions("Reply in one sentence.")},
	})
	fmt.Println("tool name:", ordersTool.Name())

	manager := openaiprovider.NewAgent(client, openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "You are the front desk. Delegate order questions to the order desk.",
		Config:       agent.Config{Name: "FrontDesk", Tools: []tool.Tool{ordersTool}},
	})

	resp, err := manager.RunText(context.Background(), "Where is order 5521?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Each call runs the specialist without a session, so it remembers nothing between calls, and its intermediate tool calls do not appear in the manager's response. `Config.RunOptions` are passed to every call; to give the specialist a memory, pass `agent.WithSession(s)` there, but then every manager run shares that one session. The tool returns only the response text (`resp.String()`). An error from the inner run becomes the tool's error, and its text reaches the manager's model ([see above](#what-the-model-sees-when-a-tool-fails)). If the specialist stops for an approval, there is no error: the manager just receives whatever text preceded the request, usually nothing, and the request is lost. Keep approval-required tools on the outer agent.

The specialist inherits the call's context, so the manager's deadline and cancellation apply. For several agents that need to share a conversation or pass control, use a [workflow](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_workflows/).

## Hosted tools

Hosted tools are markers from `tool/hostedtool`: they implement only `tool.Tool`, and the provider's service does the work. Add them to `Tools` like any other tool.

| Marker | Fields | OpenAI Responses / Foundry | Chat Completions | Anthropic | Gemini |
| --- | --- | --- | --- | --- | --- |
| `&hostedtool.WebSearch{}` | `AdditionalProperties` | `web_search` | `web_search_options` | `web_search_20250305` | Google Search |
| `&hostedtool.FileSearch{}` | `Inputs` (`HostedVectorStoreContent`), `MaximumResultCount` | `file_search` with vector store IDs; results included | skipped | skipped | file search with store names and `TopK` |
| `&hostedtool.CodeInterpreter{}` | `Inputs` (`HostedFileContent`) | `code_interpreter`, auto container with the file IDs; outputs included | skipped | `code_execution_20250825` | code execution |
| `&hostedtool.MCPServer{}` | `ServerName`, `ServerAddress`, `Authorization`, `Headers`, `AllowedTools`, `ApprovalMode` | `mcp` | skipped | skipped | skipped |

"Skipped" means the marker is not sent and no error is raised, so check the table before relying on one.

`WebSearch.AdditionalProperties` keys understood by the providers:

| Key | Value type | OpenAI Responses | Chat Completions | Anthropic |
| --- | --- | --- | --- | --- |
| `user_location` | `map[string]string` with `city`, `region`, `country`, `timezone` | yes | yes | yes |
| `search_context_size` | `string` (`low`, `medium`, `high`) | yes | yes | no |
| `filters` | `map[string]any{"allowed_domains": []string{...}}` | yes | no | no |
| `max_uses` | integer | no | no | yes |
| `allowed_domains`, `blocked_domains` | `[]string` | no | no | yes |

Results come back as content parts rather than tool results: `WebSearchToolCallContent` (with the queries), `CodeInterpreterToolCallContent` and `CodeInterpreterToolResultContent` (code, logs, generated files as `HostedFileContent` or `DataContent`), `ImageGenerationToolResultContent`, `MCPServerToolCallContent` and `MCPServerToolResultContent`. Citations are `*message.CitationAnnotation` values on the text content's `Annotations`.

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
	a := openaiprovider.NewResponsesAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "Use web search for anything current, and code for arithmetic.",
		Config: agent.Config{Tools: []tool.Tool{
			&hostedtool.WebSearch{AdditionalProperties: map[string]any{
				"user_location":       map[string]string{"country": "GB", "city": "London"},
				"search_context_size": "medium",
				"filters":             map[string]any{"allowed_domains": []string{"go.dev", "pkg.go.dev"}},
			}},
			&hostedtool.CodeInterpreter{},
			// Vector stores you have already created and filled with the OpenAI files API.
			&hostedtool.FileSearch{
				Inputs:             []message.Content{&message.HostedVectorStoreContent{VectorStoreID: "vs_handbook"}},
				MaximumResultCount: 5,
			},
		}},
	})

	resp, err := a.RunText(context.Background(),
		"What is new in the latest Go release, and what is 2^61 - 1?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)

	for c := range resp.Contents() {
		switch c := c.(type) {
		case *message.WebSearchToolCallContent:
			fmt.Println("searched:", c.Queries)
		case *message.CodeInterpreterToolCallContent:
			fmt.Println("ran code, call", c.CallID)
		case *message.CodeInterpreterToolResultContent:
			for _, out := range c.Outputs {
				if f, ok := out.(*message.HostedFileContent); ok {
					fmt.Println("generated file:", f.FileID, f.Name)
				}
			}
		}
		for _, ann := range c.Header().Annotations {
			if cite, ok := ann.(*message.CitationAnnotation); ok && cite.URL != "" {
				fmt.Println("source:", cite.Title, cite.URL)
			}
		}
	}
}
```

Hosted MCP (`hostedtool.MCPServer`) is covered on the [MCP page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_mcp/#hosted-mcp).

## The shell tool

`tool/shelltool` gives an agent a shell, modelled on .NET's `LocalShellTool`. `shelltool.NewLocal(LocalConfig{})` returns a `*shelltool.Local`, a `tool.FuncTool` called `run_shell` that takes one `command` string. It **requires approval by default**: `ApprovalRequired()` reports true unless you set `AcknowledgeUnsafe`, which you should do only when the shell already runs inside an isolation boundary such as a container.

| `LocalConfig` field | Default | Purpose |
| --- | --- | --- |
| `Name`, `Description` | `run_shell`, derived from the settings | what the model sees |
| `Shell`, `ShellArgv` | `AGENT_FRAMEWORK_SHELL`, else `/bin/bash`, or `pwsh`/`cmd` on Windows | the shell binary, or a full launch prefix; mutually exclusive |
| `Mode` | `shelltool.ModePersistent` | one long-lived shell, so `cd` and exported variables persist between calls; `ModeStateless` starts a fresh shell per command and is safe to share |
| `WorkingDirectory` | the process's | starting directory |
| `ConfineWorkingDirectory *bool` | true | in persistent mode, `cd` back to `WorkingDirectory` before each command |
| `Environment map[string]*string` | none | extra variables; a nil value removes an inherited one |
| `CleanEnvironment` | false | start from `PATH`, `HOME`, `USER`, `USERNAME`, `USERPROFILE`, `SystemRoot`, `TEMP`, `TMP` only |
| `Timeout *time.Duration` | none | per-command deadline (30 seconds is the recommended value); the result reports `TimedOut` |
| `MaxOutputBytes` | 64 KiB | per stream; beyond it output is cut and `Truncated` set |
| `Policy *shelltool.Policy` | allow all | allow and deny regular expressions, checked before the shell sees the command |
| `AcknowledgeUnsafe` | false | drop the approval requirement |

A persistent shell holds state, so never share one `Local` between users or sessions; create one per session, or use `ModeStateless`. Call `Close()` to stop the shell process.

`shelltool.NewPolicy(PolicyConfig{DenyList, AllowList, Custom})` builds the filter. Patterns are matched case-insensitively. Deny patterns are checked first; a non-nil allow list is exclusive (an empty one denies everything); `Custom` runs last and can deny a command the lists allowed, but cannot rescue one they denied. The package documentation is blunt that this is a convenience, not a security boundary: approval or a sandbox is the control.

`shelltool.NewEnvironmentProvider(executor, cfg)` is a context provider that probes the shell once (family, OS, shell version, working directory, and `--version` of a few CLIs, or `ProbeTools`) and adds instructions describing it, so the model writes commands for the shell it actually has.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/shelltool"
	"github.com/openai/openai-go/v3"
)

func main() {
	ctx := context.Background()

	policy, err := shelltool.NewPolicy(shelltool.PolicyConfig{
		DenyList: []string{`\brm\s+-rf\b`, `\bsudo\b`, `curl .*\|\s*sh`},
	})
	if err != nil {
		log.Fatal(err)
	}
	workdir, err := os.MkdirTemp("", "agent-shell")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(workdir) }()

	shell, err := shelltool.NewLocal(shelltool.LocalConfig{
		Mode:             shelltool.ModeStateless,
		WorkingDirectory: workdir,
		CleanEnvironment: true,
		Timeout:          new(30 * time.Second),
		Policy:           policy,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = shell.Close() }()

	env := shelltool.NewEnvironmentProvider(shell, shelltool.EnvironmentProviderConfig{
		ProbeTools: []string{"git", "go"},
	})

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "You are a build assistant. Use the shell to inspect the workspace.",
		Config: agent.Config{
			Tools:            []tool.Tool{shell},
			ContextProviders: []agent.ContextProvider{env},
		},
	})

	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := a.RunText(ctx, "What files are in the workspace?", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}

	// The shell requires approval; this sample approves anything the policy allows.
	var answers []message.Content
	for c := range resp.Contents() {
		if req, ok := c.(*message.ToolApprovalRequestContent); ok {
			call := req.ToolCall.(*message.FunctionCallContent)
			fmt.Println("approving", call.Arguments)
			answers = append(answers, req.CreateResponse(true, ""))
		}
	}
	if len(answers) > 0 {
		resp, err = a.RunMessage(ctx, message.New(answers...), agent.WithSession(session)).Collect()
		if err != nil {
			log.Fatal(err)
		}
	}
	fmt.Println(resp)

	// The tool can also be driven directly.
	res, err := shell.Run(ctx, "echo hello && ls")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("exit=%d timedOut=%v\n%s", res.ExitCode, res.TimedOut, res.Stdout)
	if snap, ok := env.CurrentSnapshot(); ok {
		fmt.Println("shell family:", snap.Family)
	}
}
```

A command the policy denies is not run, and `Call` returns an error wrapping `shelltool: command rejected` with the reason. Like any tool error, its text (including the reason) reaches the model, and the failure counts towards the consecutive-error limit. `Result.FormatForModel()` is what the tool returns to the model: stdout, stderr, the exit code and any timeout or truncation flags in one block.

## Differences from Python and .NET

- **No decorator or attribute.** Python's `@tool` and .NET's `AIFunctionFactory.Create` read a function's signature; Go derives the schema from a struct type, with descriptions in `jsonschema` tags.
- **Validation is strict by default.** Every struct schema has `additionalProperties: false`, and arguments are validated before the call. Python validates through Pydantic, with looser defaults.
- **The loop is middleware.** Python's `FunctionInvocationConfiguration` and .NET's `FunctionInvokingChatClient` correspond to `toolautocall.Config`, set per agent through `ToolAutoCall`. Go additionally lets `MaximumIterationsPerRequest: new(0)` switch it off.
- **Approvals** use the same request and response content types as .NET (`ToolApprovalRequestContent`, `ToolApprovalResponseContent`); Python calls them user-input requests. Standing approvals (`toolapproval`) match the .NET harness.
- **Not in Go:** tool result size limits, declaration-only tools with a separate executor (use `AdditionalTools` or the loop switched off), and a Docker shell executor. The [Python tools page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_tools/) covers its equivalents.

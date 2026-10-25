---
title: "Microsoft Agent Framework (Go) — Architecture Diagrams"
description: "Mermaid diagrams of the Go SDK as it is built: package layers, the agent run pipeline and tool loop, approvals, sessions, the workflow superstep engine, run states, checkpoint and resume, the agentworkflow chat protocol, orchestration topologies, A2A and AG-UI hosting, the telemetry span tree and a deployment layout."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "Diagrams"
  order: 17
---

These diagrams describe `github.com/microsoft/agent-framework-go` v%%version=github.com/microsoft/agent-framework-go%% as it is implemented, using the real package and type names. Each one links to the page that explains the part in detail.

## Package layers

Everything is built on two small cores: `agent` (an `*agent.Agent` wrapping a provider's run function) and `workflow` (the graph engine). Providers, tools and hosting adapters sit around them; nothing in the core depends on a particular model vendor.

```mermaid
graph TD
    subgraph app["Your application"]
        HTTP["net/http server"]
        CLI["CLI, job or test"]
    end

    subgraph hosting["Hosting adapters"]
        A2A["provider/a2aprovider<br/>NewHandler / NewAgentFromResolver"]
        AGUI["provider/aguiprovider<br/>NewJSONHTTPHandler / NewAgent"]
        MCPT["tool/mcptool<br/>AddTool / Connect"]
    end

    subgraph orchestration["Orchestration"]
        AW["workflow/agentworkflow<br/>sequential, concurrent, group chat, NewAgent"]
        WF["workflow + workflow/inproc<br/>Builder, Executor, environments"]
        CP["workflow/checkpoint<br/>Manager, Store"]
    end

    subgraph core["Agent core"]
        AG["agent<br/>Agent, Session, Middleware, ContextProvider"]
        MSG["message<br/>Message, Contents"]
        TOOLS["tool, tool/functool, tool/agenttool"]
        HARN["agent/harness/toolautocall<br/>the tool loop"]
    end

    subgraph providers["Model providers"]
        OAI["openaiprovider"]
        FDY["foundryprovider"]
        ANT["anthropicprovider"]
        GEM["geminiprovider"]
        COP["copilotprovider"]
    end

    OTEL["provider/otelprovider<br/>workflow/observability/opentelemetry"]

    HTTP --> A2A
    HTTP --> AGUI
    HTTP --> MCPT
    CLI --> AG
    CLI --> WF
    A2A --> AG
    AGUI --> AG
    MCPT --> TOOLS
    AW --> WF
    AW --> AG
    WF --> CP
    AG --> MSG
    AG --> TOOLS
    AG --> HARN
    OAI --> AG
    FDY --> AG
    ANT --> AG
    GEM --> AG
    COP --> AG
    OTEL -.-> AG
    OTEL -.-> WF
```

Details: [comprehensive guide](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_comprehensive_guide/), [providers](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_providers/).

## The agent run pipeline

A run passes through two layers of middleware. `agent.Config.Middlewares` wrap the whole run, outside history and context providers. `agent.ProviderConfig.Middlewares` wrap the provider's run function; the provider constructors install the tool loop there. Function middleware wraps each tool call.

```mermaid
graph LR
    RT["a.RunText(ctx, text, opts...)"] --> AM["Agent middlewares<br/>otelprovider, logging, retries"]
    AM --> HP["History provider<br/>load session history"]
    HP --> CPV["Context providers<br/>instructions, memory, RAG"]
    CPV --> PM["Provider middlewares<br/>toolautocall"]
    PM --> PR["Provider Run<br/>OpenAI, Foundry, Anthropic, Gemini"]
    PR --> API[("Model API")]
    PM -->|"function call"| FM["Function middlewares"]
    FM --> FT["tool.FuncTool.Call"]
    FT -->|"FunctionResultContent"| PM
    PR -->|"ResponseUpdate stream"| PM
    PM --> CPV
    CPV -->|"store new messages"| HP
    HP --> AM
    AM -->|"iter.Seq2 of updates"| RT
```

Details: [middleware](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_middleware/#the-run-pipeline), [tools](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_tools/#the-tool-loop).

## The tool loop

`toolautocall` turns `FunctionCallContent` into tool invocations and calls the model again, up to `MaximumIterationsPerRequest` rounds (40 by default). When the limit is reached it makes one last call without tools.

```mermaid
sequenceDiagram
    participant App as Your code
    participant Auto as toolautocall
    participant Model as Provider and model
    participant Tool as functool tool

    App->>Auto: Run(messages)
    Auto->>Model: messages + tool schemas
    Model-->>Auto: FunctionCallContent(get_weather)
    Auto->>Tool: Call(ctx, arguments JSON)
    Tool-->>Auto: result or error
    Auto->>Model: messages + FunctionResultContent
    Model-->>Auto: TextContent
    Auto-->>App: ResponseUpdates (call, result, text)
    Note over Auto: failing rounds count towards MaximumConsecutiveErrorsPerRequest, 3 by default
```

## Tool approvals

A tool wrapped in `tool.ApprovalRequiredFunc` is not called. The run ends with a `ToolApprovalRequestContent` in the response, and the next run on the same session carries the decision.

```mermaid
sequenceDiagram
    participant User
    participant App as Your code
    participant Agent as *agent.Agent
    participant Model

    App->>Agent: RunText("refund A-77", WithSession(s))
    Agent->>Model: request
    Model-->>Agent: FunctionCallContent(issue_refund)
    Agent-->>App: ToolApprovalRequestContent
    App->>User: approve issue_refund?
    User-->>App: yes
    App->>Agent: RunMessage(req.CreateResponse(true, reason), WithSession(s))
    Agent->>Agent: call issue_refund
    Agent->>Model: FunctionResultContent
    Model-->>Agent: final text
    Agent-->>App: response
```

Details: [tools](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_tools/#approvals).

## Sessions and history

A session is a JSON-serialisable state bag plus an optional service ID. Where the conversation itself lives depends on the provider.

```mermaid
graph TD
    S["*agent.Session"]
    S --> BAG["State bag<br/>Set / Get, provider and context-provider state"]
    S --> SID["ServiceID<br/>response or conversation ID"]
    BAG --> LH["In-memory history<br/>Chat Completions, Anthropic, Gemini,<br/>Responses with DisableStoreOutput"]
    SID --> SH[("Service-held history<br/>OpenAI Responses, Foundry")]
    S -->|"json.Marshal"| DB[("Your store")]
    DB -->|"json.Unmarshal"| S
```

Details: [sessions and memory](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_sessions_and_memory/).

## Workflow building blocks

The builder stores bindings, not executors. A binding either shares one executor instance between runs (`Bind`) or makes one per run (`BindNewExecutorFunc`).

```mermaid
graph LR
    F["func(T) U"] --> NE["workflow.NewExecutor(id, v)"]
    ST["struct with Handle(ctx, T) error"] --> NE
    RAW["*workflow.Executor<br/>ConfigureProtocol + RouteBuilder"] --> B1
    NE --> B1["executor.Bind()<br/>shared instance"]
    FAC["factory(sessionID, id)"] --> B2["workflow.BindNewExecutorFunc<br/>instance per run"]
    RP["workflow.RequestPort"] --> B3["port.Bind()"]
    SUB["*workflow.Workflow"] --> B4["inproc.BindSubworkflowAsExecutor"]
    AGT["*agent.Agent"] --> B5["agentworkflow.New"]
    B1 --> BLD["workflow.NewBuilder(start)<br/>AddEdge, AddFanOutEdge, AddFanInBarrierEdge,<br/>AddSwitch, WithOutputFrom"]
    B2 --> BLD
    B3 --> BLD
    B4 --> BLD
    B5 --> BLD
    BLD -->|"Build: type-checks every edge"| W["*workflow.Workflow"]
    W --> ENV["inproc.Default / Concurrent / Lockstep<br/>optional WithCheckpointing"]
```

Details: [workflows](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_workflows/#executors).

## Supersteps

Messages sent in one superstep are delivered at the start of the next. Executors that receive messages in the same superstep run concurrently. State writes are applied at the end of the superstep, and a checkpoint, if enabled, is taken then.

```mermaid
sequenceDiagram
    participant R as Runner (inproc)
    participant A as Executor A
    participant B as Executor B
    participant C as Executor C
    participant St as State and checkpoint

    Note over R: superstep 0
    R->>A: deliver input
    A-->>R: SendMessage(x) to B and C (fan-out)
    R->>St: apply queued state updates
    R->>St: checkpoint (if enabled)
    Note over R: superstep 1
    par concurrently
        R->>B: deliver x
        R->>C: deliver x
    end
    B-->>R: SendMessage(y)
    C-->>R: YieldOutput(z)
    R-->>R: OutputEvent(z)
    R->>St: apply updates, checkpoint
    Note over R: superstep 2 ... until no messages remain
```

## Run states

`inproc.RunStatus` as a run moves between halts. A halted run accepts more input; one waiting on requests accepts responses.

```mermaid
stateDiagram-v2
    [*] --> NotStarted
    NotStarted --> Running: Run / RunStreaming / TrySendMessage
    Running --> Idle: no messages left
    Running --> PendingRequests: RequestInfoEvent outstanding
    Idle --> Running: TrySendMessage / Resume(msgs)
    PendingRequests --> Running: SendResponse / Resume(resp)
    Running --> Ended: executor failed or ErrorEvent
    Running --> Ended: CancelRun / context cancelled
    Idle --> Ended: Close
    PendingRequests --> Ended: Close (resume later from a checkpoint)
    Ended --> [*]
```

## Human in the loop with a checkpoint

The approval below takes days, so the first process stops after the request is raised. Another process resumes from the latest checkpoint into a freshly built, identical graph; the pending request is re-emitted and answered.

```mermaid
sequenceDiagram
    participant P1 as Process 1
    participant Run1 as StreamingRun
    participant Store as checkpoint.Store
    participant P2 as Process 2
    participant Run2 as Resumed run
    participant Person

    P1->>Run1: RunStreaming(wf, refund, WithSessionID("EX-2"))
    Run1->>Store: checkpoint after each superstep
    Run1-->>P1: RequestInfoEvent(manager approval)
    P1->>Run1: Close (WatchUntilHalt returned)
    Note over Store: days pass
    Person->>P2: POST /expenses/EX-2/decision
    P2->>Store: LatestCheckpoint("EX-2")
    P2->>Run2: ResumeStreaming(buildWorkflow(), checkpoint)
    Run2-->>P2: RequestInfoEvent (re-emitted)
    P2->>Run2: SendResponse(request.CreateResponse(decision))
    Run2->>Store: checkpoint
    Run2-->>P2: OutputEvent(outcome)
```

Details: [checkpointing](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_workflows/#checkpointing), [the expense recipe](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_recipes/#expense-approvals-over-http).

## Agents as executors: the turn-token protocol

An executor made by `agentworkflow.New` collects chat messages until a `workflow.TurnToken` arrives, runs the agent once, and passes the conversation and the token on. Tool approvals inside the turn become external requests.

```mermaid
sequenceDiagram
    participant Up as Upstream executor
    participant H as agentworkflow.New(agent)
    participant Ag as *agent.Agent
    participant Down as Downstream executor
    participant Caller

    Up->>H: *message.Message / []*message.Message
    Note over H: collected, nothing runs yet
    Up->>H: TurnToken{EmitEvents: true}
    H->>Ag: Run(collected messages)
    Ag-->>H: ResponseUpdates
    H-->>Caller: OutputEvent(*agent.ResponseUpdate) if registered as output
    alt approval needed
        H-->>Caller: RequestInfoEvent(ToolApprovalRequestContent)
        Caller->>H: SendResponse(approval)
        H->>Ag: Run again with the decision
    end
    H->>Down: []*message.Message (conversation)
    H->>Down: TurnToken
```

Details: [agents in workflows](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_workflows/#agents-in-workflows).

## Orchestration topologies

The three prebuilt patterns in `agentworkflow`, as graphs.

```mermaid
graph LR
    subgraph seq["NewSequentialWorkflowBuilder"]
        S0(["input"]) --> S1["agent 1"] --> S2["agent 2"] --> S3["agent 3"] --> SO(["conversation"])
    end
```

```mermaid
graph LR
    subgraph con["NewConcurrentWorkflowBuilder"]
        C0(["input"]) --> C1["agent 1"]
        C0 --> C2["agent 2"]
        C0 --> C3["agent 3"]
        C1 --> AGG["aggregator<br/>default or WithAggregator"]
        C2 --> AGG
        C3 --> AGG
        AGG --> CO(["messages"])
    end
```

```mermaid
graph TD
    subgraph gc["NewGroupChatWorkflowBuilder"]
        G0(["input"]) --> MGR["GroupChatManager<br/>SelectNextAgent, ShouldTerminate"]
        MGR -->|"next speaker"| P1["agent 1"]
        MGR -->|"next speaker"| P2["agent 2"]
        P1 -->|"reply"| MGR
        P2 -->|"reply"| MGR
        MGR -->|"nil or terminate"| GO(["conversation"])
    end
```

Details: [orchestrations](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_orchestrations/).

## A workflow as an agent

`agentworkflow.NewAgent` puts a workflow behind the `*agent.Agent` interface. The first run on a session opens a streaming workflow run, and later runs on that session feed it.

```mermaid
graph LR
    CALLER["Caller<br/>RunText, A2A, AG-UI, agenttool"] --> WA["agentworkflow.NewAgent(wf)"]
    WA -->|"first run on a session"| OPEN["RunStreaming"]
    WA -->|"later runs"| SEND["TrySendMessage / SendResponse"]
    OPEN --> RUN["StreamingRun"]
    SEND --> RUN
    RUN -->|"hosted agents' updates"| WA
    RUN -->|"RequestInfoEvent"| REQ["ToolApprovalRequestContent<br/>in the response"]
    REQ --> CALLER
    WA -->|"ResponseUpdates"| CALLER
```

## A2A hosting and tasks

The A2A handler keeps one session per context ID in its `SessionStore`. In task mode, requests become tasks that a client can poll or resubscribe to.

```mermaid
graph LR
    CL["a2aprovider.NewAgentFromResolver"] -->|"GET /.well-known/agent-card.json"| CARD["NewStaticAgentCardHandler"]
    CL -->|"POST / (JSON-RPC) or REST"| AUTH["your auth middleware"]
    AUTH --> TR["a2asrv.NewJSONRPCHandler / NewRESTHandler"]
    TR --> RH["a2aprovider.NewHandler"]
    RH --> SS[("SessionStore<br/>by context ID")]
    RH --> TS[("Task store<br/>optionally isolation-key scoped")]
    RH --> AG["*agent.Agent or workflow agent"]
```

```mermaid
stateDiagram-v2
    [*] --> Submitted: ReturnTask, background request
    Submitted --> Working
    Working --> InputRequired: agent needs input or approval
    InputRequired --> Working: next message linked to the task
    Working --> Completed
    Working --> Failed
    Completed --> [*]
    Failed --> [*]
```

Details: [hosting and A2A](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_hosting_and_a2a/).

## AG-UI

The AG-UI handler is stateless: the client sends the whole conversation and its own tool declarations with every run, and receives Server-Sent Events.

```mermaid
sequenceDiagram
    participant UI as Front end or aguiprovider.NewAgent
    participant H as aguiprovider.NewJSONHTTPHandler
    participant Ag as *agent.Agent

    UI->>H: POST run input (thread, messages, client tools)
    H->>Ag: Run(messages, WithTool(declarations))
    H-->>UI: RUN_STARTED
    Ag-->>H: text, tool calls, results
    H-->>UI: TEXT_MESSAGE events, TOOL_CALL events
    H-->>UI: RUN_FINISHED
    Note over UI: runs client tool calls itself,<br/>then sends the results in the next request
```

## Telemetry

What one agent run with a tool call looks like in a trace, inside a workflow executor and an incoming HTTP request.

```mermaid
graph TD
    REQ["HTTP server span<br/>otelhttp"] --> WI["workflow_invoke"]
    WI --> EX["executor.process summarise"]
    EX --> IA["invoke_agent Forecaster"]
    IA --> C1["chat model<br/>finish_reasons=tool_calls"]
    IA --> T1["execute_tool get_weather"]
    IA --> C2["chat model<br/>finish_reasons=stop"]
    WS["workflow.session"]
    WB["workflow.build"]
```

`workflow.build` and `workflow.session` are separate spans, not parents of the run. Details: [observability](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_observability/).

## A deployment

One way to run the pieces: a stateless Go service, scaled out, with conversation and workflow state in shared stores and telemetry through a collector.

```mermaid
graph LR
    U["Users and other agents"] --> LB["Load balancer / gateway<br/>TLS, auth"]
    LB --> P1["Go service replica 1"]
    LB --> P2["Go service replica 2"]
    P1 --> DB[("Sessions and checkpoints<br/>database")]
    P2 --> DB
    P1 --> M[("Model provider<br/>OpenAI, Azure, Anthropic, Gemini")]
    P2 --> M
    P1 -.->|"managed identity"| KV["Key Vault / secrets"]
    P1 -.->|"OTLP"| COL["OpenTelemetry Collector"]
    P2 -.->|"OTLP"| COL
    COL --> OBS["Tracing and metrics backend"]
    P1 -->|"A2A"| REMOTE["Remote agents"]
```

Details: [production](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_production/).

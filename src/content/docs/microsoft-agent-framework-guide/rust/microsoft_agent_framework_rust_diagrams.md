---
title: "Microsoft Agent Framework (Rust) — Diagrams"
description: "Diagrams of how agent-framework-rs works, drawn from its source: the crate map, the agent run pipeline and where each kind of middleware sits, the tool loop and approval round trip, session and history flow, the workflow superstep engine, each orchestration, the hosting surfaces and the MCP client lifecycle."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Diagrams"
  order: 16
---

These diagrams show how `agent-framework` %%version=agent-framework-rs%% works inside, drawn from its source rather than from the Python SDK's documentation. Where the port behaves differently from Python, the diagram shows the port. Each diagram has a short explanation under it and a link to the page that covers the topic in full.

The diagrams are Mermaid source. Paste one into any Mermaid renderer (GitHub Markdown, the Mermaid live editor, most IDE previews) to see it drawn.

## Crates

```mermaid
graph LR
    facade["agent-framework<br/>(facade, prelude, features)"]
    core["agent-framework-core<br/>agents, tools, sessions, middleware,<br/>workflows, orchestrations, observability"]

    facade --> core
    facade -. "feature" .-> openai & anthropic & azure & foundry & gemini & mistral & ollama & flocal & bedrock & copilot & mcp & a2a & hosting & declarative & redis & mem0 & cosmos & search & cstudio & purview & tools

    openai["openai"] --> core
    gemini["gemini"] --> core
    bedrock["bedrock"] --> core
    a2a["a2a (client)"] --> core
    mcp["mcp"] --> core
    hosting["hosting"] --> core
    declarative["declarative"] --> core
    redis["redis"] --> core
    mem0["mem0"] --> core
    cstudio["copilotstudio"] --> core
    purview["purview"] --> core
    tools["tools (shell)"] --> core

    azure["azure"] --> core
    azure --> openai
    ollama["ollama"] --> openai
    mistral["mistral"] --> openai
    flocal["foundry-local"] --> openai
    copilot["github-copilot"] --> openai
    foundry["foundry"] --> azure
    foundry --> openai
    anthropic["anthropic"] --> bedrock
    anthropic --> azure
    cosmos["cosmos"] --> azure
    search["azure-ai-search"] --> azure
```

Every companion crate depends on `agent-framework-core`, which has no dependency on any of them; the arrows above are the extra dependencies between companions (each node is `agent-framework-<name>`, and every one also points at core). The facade crate `agent-framework` re-exports core at its root and each companion as a module behind a cargo feature of the same name, so `features = ["anthropic"]` gives `agent_framework::anthropic`. Several providers reuse the OpenAI crate's wire code: Ollama, Mistral, Foundry Local and GitHub Copilot speak Chat Completions, and Azure and Foundry wrap the OpenAI clients with Azure authentication. `anthropic` pulls in `bedrock` and `azure` for Claude on Bedrock and Foundry. `hosting` depends only on core: its A2A server has its own copy of the wire types and does not use the `a2a` client crate. The [overview](/microsoft-agent-framework-guide/rust/#cargo-features) lists the features.

## The agent run pipeline

```mermaid
flowchart TD
    entry["Agent::run / run_with_options<br/>input, optional session, AgentRunOptions"]
    entry --> session{"session given?"}
    session -- no --> fresh["create_session()"]
    session -- yes --> ensure
    fresh --> ensure["ensure_history_provider:<br/>no service id and no history provider<br/>→ insert InMemoryHistoryProvider first"]
    ensure --> opts["merge ChatOptions: agent defaults ← per-run overrides<br/>conversation_id = session service id, else option"]
    opts --> before["before_run on every provider, in order:<br/>session's providers, then the agent's<br/>(history providers skipped when the service holds history)"]
    before --> tools["tools = agent tools + provider tools + per-run tools<br/>+ ToolSource::resolve_tools() (deduplicated by name)"]
    tools --> msgs["messages = system(instructions + provider instructions)<br/>+ provider messages (history) + input"]
    msgs --> amw["agent middleware chain<br/>(invoke_agent span around it)"]
    amw --> cmw["chat middleware chain<br/>(once per run)"]
    cmw --> fic["FunctionInvokingChatClient<br/>the tool loop + function middleware"]
    fic --> deco["your ChatClient decorators<br/>RetryingChatClient, ObservableChatClient, ...<br/>(once per model request)"]
    deco --> provider["provider client → HTTP"]
    provider -. "response, back up through each layer" .-> cid{"response conversation_id?"}
    cid -- "none, but session is service-held" --> err["Error::AgentExecution"]
    cid -- "some" --> adopt["session adopts the id"]
    cid -- "none, local session" --> after
    adopt --> after["after_run(input, response) on every provider<br/>(history providers skipped if the service holds history)"]
    after --> done["AgentResponse"]
```

A run first assembles the request, then sends it down a fixed stack of layers, and finally records the result. `prepare_request` attaches an in-memory history provider to a local session that has none, merges the per-run `ChatOptions` over the agent's (the per-run side wins), and runs every context provider's `before_run` over one shared `SessionContext`: the session's providers first, then the agent's own (`with_compaction` registers an agent provider, so compaction sees the history already loaded). Provider instructions are appended after the agent's, provider messages go before the caller's input, and tools from providers, from `AgentRunOptions` and from tool sources (resolved fresh on every run) are added unless the name is taken.

The stack below is fixed at build time. `AgentBuilder::build` wraps the client you passed in a `FunctionInvokingChatClient`, so agent middleware runs once per run, chat middleware runs once per run around the whole tool loop (Python runs chat middleware once per model request), function middleware runs once per tool call, and anything you wrapped around the provider client before `Agent::builder` runs once per model request. On success the session adopts any conversation id the service returned and each provider's `after_run` records the original input and the response; on failure each provider's `after_run` sees the error, its own result is discarded, and the run's error is returned. The [middleware page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_middleware/#three-levels) and [context and memory](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_context_and_memory/#where-providers-attach-and-the-order-they-run-in) cover the two halves.

### Where agent hooks attach

```mermaid
flowchart LR
    ah["agent hook seam<br/>(first agent middleware)"] --> amw["your agent middleware"] --> cmw["chat middleware"] --> fic["tool loop"]
    fic --> fo["function hook seam<br/>(first function middleware)"] --> fmw["your function middleware"] --> fi["function hook seam<br/>(last function middleware)"] --> tool["tool"]
    fic --> mh["model-call hook seam<br/>(wraps the raw client)"] --> client["provider client"]
```

With `experimental-agent-hooks`, `AgentHooks` builds the agent with its seams inserted: one outermost agent middleware, function middleware at both ends of your chain (so it sees the arguments before and after your middleware changes them), and a decorator directly around the raw client, below the tool loop. See [agent hooks](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_middleware/#agent-hooks-experimental).

## Streaming runs

```mermaid
flowchart TD
    rs["Agent::run_stream"] --> prep["prepare_request (as for run)"]
    prep --> has{"agent middleware configured?"}
    has -- yes --> core["run the full pipeline as run() does"]
    core --> replay["adopt conversation id, after_run on providers,<br/>then replay the response's messages as updates"]
    has -- no --> pre["chat middleware: pre-call changes only<br/>(no post-processing, no short-circuit)"]
    pre --> tools{"executable tools?"}
    tools -- no --> tokens["provider stream: token-level updates"]
    tools -- yes --> whole["tool loop runs to completion,<br/>then whole messages are replayed"]
    tokens --> end_["stream end: adopt conversation id,<br/>after_run on providers"]
    whole --> end_
    replay --> out["AgentRunStream"]
    end_ --> out
```

Token-level streaming only happens when there is no agent middleware and no local tool. With agent middleware, `run_stream` goes through the same pipeline as `run` so guardrails apply, records the run, and only then replays the result as updates; with local tools, the tool loop's streaming method runs the whole loop and replays its messages, carrying usage on the last update. Chat middleware on the plain token path can only change the request, because a live stream cannot pass back through `ChatContext::result`. On the other two paths, context providers record the run when the stream ends, so a stream dropped before the end records nothing. See [streaming runs](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_middleware/#streaming-runs).

## The tool loop

```mermaid
flowchart TD
    start["FunctionInvokingChatClient::get_response"] --> v["validate config, take the session from the options,<br/>tool_choice = Auto if tools and none set"]
    v --> any{"executable tools and loop enabled?"}
    any -- no --> single["one model request"] --> ret
    any -- yes --> budget["resume the budget parked in the session (if this request<br/>carries approval answers), else start a new one"]
    budget --> iter["iteration (up to max_iterations, default 40)"]
    iter --> appr{"unanswered approval responses<br/>in the conversation?"}
    appr -- yes --> runappr["run approved calls, reject the others<br/>(result: 'Tool call invocation was rejected by user.')"]
    appr -- no --> spent
    runappr --> spent{"budget spent?<br/>max_function_calls / max_duration_seconds"}
    spent -- yes --> final["final request with tool_choice = None"]
    spent -- no --> req["model request, add its usage to the total"]
    req --> calls{"unresolved function calls?"}
    calls -- no --> ret["return carried messages + reply,<br/>usage summed over every request"]
    calls -- yes --> needs{"any call to a tool<br/>requiring approval?"}
    needs -- yes --> pause["append FunctionApprovalRequestContent per call,<br/>park the budget in the session, return"]
    needs -- no --> decl{"declaration-only tool called?"}
    decl -- yes --> back["run the local calls,<br/>return the rest to the caller"]
    decl -- no --> exec["run calls through function middleware<br/>(concurrently by default)"]
    exec --> errs{"more than max_consecutive_errors_per_request (3)<br/>failing batches in a row?"}
    errs -- yes --> nomore["tool_choice = None for later requests"] --> next
    errs -- no --> next["append the tool results<br/>(only the results if the service holds the conversation)"]
    next --> iter
    iter -- "max_iterations reached" --> final
    final --> ret
```

The loop sits below chat middleware and above your client decorators. Every model request goes through it; a reply with no unresolved function calls ends it. The two budgets are graceful: when either is spent the loop sends one last request with tools disabled, so the model answers with what it has, and they are checked between batches so a parallel batch always completes. An unknown tool name gets an error result ("tool 'x' not found") unless `terminate_on_unknown_calls` makes it fail the run; a tool error becomes an error result whose text is "tool execution failed" unless `include_detailed_errors` is set; a middleware failure propagates and fails the run. When the service holds the conversation (a reply carried a conversation id), each later request sends only the new tool results. The [tool loop](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/#the-tool-loop) section gives the limits and their defaults.

## The approval round trip

```mermaid
sequenceDiagram
    participant App
    participant Agent
    participant TL as Tool loop
    participant Model
    participant Tool

    App->>Agent: run([user msg], session)
    Agent->>TL: get_response(history + input)
    TL->>Model: request with tools
    Model-->>TL: FunctionCall(delete_file)
    Note over TL: delete_file requires approval:<br/>nothing runs, budget parked in session
    TL-->>Agent: assistant msg: call + FunctionApprovalRequestContent
    Agent-->>App: response.user_input_requests() = [request]
    Note over Agent: history records the call and the request
    App->>Agent: run([user msg: request.create_response(true)], session)
    Agent->>TL: get_response(history + approval response)
    Note over TL: budget resumed: same count and clock
    TL->>Tool: invoke (through function middleware)
    Tool-->>TL: result
    TL->>Model: history with the call and its result
    Model-->>TL: final text
    TL-->>Agent: tool result + answer
    Agent-->>App: response.text()
```

A tool marked `ApprovalMode::AlwaysRequire` is never run on the model's say-so. If any call in a batch needs approval, the loop returns the whole batch as approval requests, each with its own occurrence id, and the session's history records them. The caller answers on the same session with `create_response(true)` or `create_response(false)`; the loop finds the unanswered responses in the conversation, runs the approved calls (a rejected one gets the rejection text as its result), rewrites the request and response pair as an ordinary call and result, and carries on. The call and time budget survive the round trip because they are parked in the session's state, so approve-and-continue cannot reset them. See [approvals](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/#the-approval-round-trip).

## Sessions and history

```mermaid
sequenceDiagram
    participant App
    participant Agent
    participant Hist as InMemoryHistoryProvider
    participant Svc as Model service

    Note over App,Svc: Local history (Chat Completions and other stateless APIs)
    App->>Agent: run(input, session)
    Agent->>Hist: before_run: prepend stored messages
    Agent->>Svc: system + history + input
    Svc-->>Agent: reply (no conversation id)
    Agent->>Hist: after_run: store input + reply

    Note over App,Svc: Service-held history (OpenAI Responses API, Foundry)
    App->>Agent: run(input, session)
    Agent->>Hist: before_run (first turn: nothing stored)
    Agent->>Svc: system + input
    Svc-->>Agent: reply + conversation id
    Note over Agent: session adopts the id as its service_session_id
    Note over Agent,Hist: after_run skips history providers: the service has the turn
    App->>Agent: run(next input, session)
    Note over Agent,Hist: history providers skipped (service id present)
    Agent->>Svc: system + next input + conversation_id
    Svc-->>Agent: reply (same or new id, the session adopts it)
```

An `AgentSession` holds an id, an optional service-side conversation id, a state bag shared by its clones, and its own list of context providers, one of which is normally a history provider. With a stateless API the in-memory provider (attached automatically on the first run) sends the whole transcript every time and records each turn. When the service returns a conversation id, the session adopts it and from then on history providers are left out of both `before_run` and `after_run`, as .NET does, so turns are not sent twice; a session created with `AgentSession::service(id)` or `create_session_with_service_id` starts in that mode. A service-held session whose reply carries no id fails with `Error::AgentExecution`. `to_dict` saves the ids and the state bag but not the history, which lives in the provider. See [service-held and local history](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_sessions/#service-held-and-local-history).

## The workflow engine

```mermaid
flowchart TD
    run["Workflow::run(input) / run_stream(input)"] --> q0["queue = [message → start executor]<br/>events: Started, Status(InProgress)"]
    q0 --> loop{"queue empty?"}
    loop -- no --> limit{"iteration ≥ max_iterations?"}
    limit -- yes --> fail["Error: max_iterations exceeded"]
    limit -- no --> route["SuperStepStarted(n)<br/>route each queued message: direct target,<br/>or edge groups (conditions evaluated now)"]
    route --> fanin["fan-in targets: buffer per source<br/>until every source has sent"]
    fanin --> exec["run targets in parallel (join_all),<br/>several messages for one target run in turn"]
    exec --> apply["apply effects in sorted executor-id order:<br/>outputs, custom events, requests, sent messages"]
    apply --> ok{"any executor failed?"}
    ok -- yes --> discard["discard staged shared state<br/>ExecutorFailed, Failed, Status(Failed)"]
    ok -- no --> commit["SharedState::commit()<br/>SuperStepCompleted(n)"]
    commit --> cp["checkpoint (if storage configured)"]
    cp --> loop
    loop -- yes --> idle{"pending requests?"}
    idle -- no --> s1["Status(Idle)"]
    idle -- yes --> s2["Status(IdleWithPendingRequests)"]
```

The engine is a Pregel-style loop of supersteps. Messages sent during one superstep are delivered in the next, after routing through the edges, so a condition or switch is evaluated when the message is routed, not when it is sent. All targets of a superstep run concurrently; their effects are then applied in sorted executor-id order, which is why outputs come out in id order. A fan-in target is invoked once, with a JSON array in declared source order, when every source has delivered. A failing executor fails the whole run and its superstep's shared-state writes are discarded. `max_iterations` counts supersteps (100 by default), so a loop in the graph needs a condition that stops it. See [workflows](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/#concurrency-within-a-superstep).

### Shared state within a superstep

```mermaid
sequenceDiagram
    participant A as Executor A
    participant B as Executor B
    participant S as SharedState
    participant R as Runner

    Note over A,B: superstep n (A and B run concurrently)
    A->>S: set("count", 1)  (staged, pending)
    A->>S: get("count") → 1  (reads pending first)
    B->>S: get("count") → may see 1: same store, one RwLock
    R->>S: commit() after every executor finished
    Note over S: pending folded into committed
    R->>R: checkpoint exports committed state only
    Note over A,B: if any executor failed instead, R calls discard()
```

Shared state is one store per run, cloned into every executor's context. Writes are staged and folded into committed state only at the end of a successful superstep, so a checkpoint never holds a half-finished superstep. Reads check the staged writes first, so an executor sees its own writes; executors in the same superstep share the store, so do not rely on either seeing or not seeing the other's writes. Keys starting with `_` are reserved, and a sub-workflow gets its own store. See [shared state](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/#shared-state).

### Human in the loop and checkpoints

```mermaid
sequenceDiagram
    participant App
    participant Run as WorkflowRun
    participant Ex as Executor
    participant Store as CheckpointStorage

    App->>Run: workflow.run(input)
    Run->>Ex: execute(message, ctx)
    Ex->>Run: ctx.request_info(data)
    Run-->>App: RequestInfo { request_id, data }
    Run->>Store: save checkpoint (queue, executor states,<br/>committed state, pending requests, fan-in buffers, graph signature)
    Run-->>App: state = IdleWithPendingRequests
    Note over App: later, perhaps in another process
    App->>Run: workflow.run_from_checkpoint(id, storage)
    Run->>Store: load, compare graph signature
    Run-->>App: pending RequestInfo events again
    App->>Run: send_response(request_id, answer)
    Run->>Ex: RequestResponse { request_id, data, original_request }<br/>(to the executor that asked, next superstep)
    Ex->>Run: yield_output(...)
    Run-->>App: Output, Status(Idle)
```

`ctx.request_info` records a request with a fresh id. The run carries on while other messages are queued, then stops as `IdleWithPendingRequests`. `send_response` (or `send_responses` for several) queues a `RequestResponse` to the executor that asked and drives the run again; an unknown request id is an error. With checkpointing on, a checkpoint is saved after every superstep, including the one that raised the request. `run_from_checkpoint` refuses a checkpoint whose graph signature differs from the workflow's (`run_from_checkpoint_unchecked` skips that check), restores executor state, shared state, the queue and pending requests, and re-emits the pending `RequestInfo` events. A failed checkpoint save is logged, not raised. See [human in the loop](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/#human-in-the-loop) and [checkpointing](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/#checkpointing).

### Sub-workflows and workflows as agents

```mermaid
flowchart LR
    subgraph outer["outer workflow"]
        a["executor"] --> wx["WorkflowExecutor"] --> b["executor"]
    end
    subgraph inner["inner workflow (own run, own SharedState)"]
        s["start"] --> e["..."]
    end
    wx -- "message → inner.run" --> s
    e -- "outputs → sent on (or yielded with with_direct_output)" --> wx
    e -. "requests forwarded under their original ids" .-> wx
    wa["WorkflowAgent / workflow.as_agent()"] -- "SupportsAgentRun" --> outer
```

A `WorkflowExecutor` runs another workflow as one node: each message starts an inner run with its own shared state, the inner run's outputs are sent on from the node, and its requests surface from the outer run under their original ids so `send_response` reaches them. `WorkflowAgent` goes the other way and presents a workflow as an agent, so it can be hosted or used as a tool. See [sub-workflows](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/#sub-workflows).

## Orchestrations

Sequential and concurrent orchestrations build graphs of agent executors. Group chat, handoff and Magentic each build a workflow with a single executor that calls its participants in a loop, so their events all carry one executor id and a whole run is one superstep (or one per human answer).

### Sequential

```mermaid
flowchart LR
    inp(["input"]) --> s0["seq_0<br/>AgentExecutor"] --> s1["seq_1"] --> s2["seq_2"] --> out(["Output: full conversation"])
    s0 -. "AgentRunUpdate / AgentRun events" .-> ev[["events"]]
    s1 -.-> ev
    s2 -.-> ev
```

Each executor runs its agent over the whole conversation so far, appends the reply and sends the conversation on; only the last yields it, unless `output_from` or `intermediate_output_from` names other participants. With `with_request_info()` every node is an `AgentApprovalExecutor` instead (below). See [sequential](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/#sequential).

### Concurrent

```mermaid
flowchart LR
    inp(["input"]) --> d["dispatch"]
    d --> a0["agent_0"] & a1["agent_1"] & a2["agent_2"]
    a0 & a1 & a2 --> agg["aggregate<br/>(fan-in)"]
    agg --> out(["Output: prompt + each agent's last message"])
```

`dispatch` fans the input out, the agents run in the same superstep, and the fan-in delivers their conversations to `aggregate` in participant order. The aggregator keeps the first conversation minus its last message, then the last message of each participant; there is no custom aggregator hook. See [concurrent](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/#concurrent).

### Group chat

```mermaid
flowchart TD
    inp(["input conversation"]) --> lim{"rounds ≥ max_rounds?"}
    lim -- yes --> halt1["append 'Conversation halted after reaching manager round limit.'"] --> out
    lim -- no --> term{"termination_condition?"}
    term -- yes --> halt2["append 'Conversation halted after termination condition was met.'"] --> out
    term -- no --> mgr["manager.select(GroupChatState)"]
    mgr -- "Finish" --> fin["append final message<br/>(default 'Conversation completed.')"] --> out
    mgr -- "Speak(name, instruction)" --> instr["append instruction as manager message (if any)"]
    instr --> run["run participant over the conversation,<br/>append its reply, round += 1"]
    run --> lim
    out(["Output: conversation"])
```

Everything happens inside `group_chat_orchestrator`. The round limit is checked first, then the termination condition, then the manager decides; an unknown participant name fails the run. The manager can be round robin (the default), a closure, your own `GroupChatManager` or an LLM agent. See [group chat](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/#group-chat).

### Handoff

```mermaid
flowchart TD
    inp(["user input"]) --> start["current = initial agent"]
    start --> lim{"turns over max_iterations?"}
    lim -- yes --> out
    lim -- no --> run["run current agent over the conversation"]
    run --> det{"handoff call in the reply?"}
    det -- "known target" --> ack["append tool result {handoff_to}, current = target"] --> lim
    det -- "unknown or not allowed" --> err["append error tool result, same agent again"] --> lim
    det -- none --> term{"termination_condition?"}
    term -- yes --> out(["Output: conversation without tool plumbing"])
    term -- no --> mode{"interaction mode"}
    mode -- Autonomous --> out
    mode -- HumanInLoop --> save["save conversation in shared state,<br/>request_info(HandoffUserInputRequest)"]
    save -. "send_response(text)" .-> resume["append as user message,<br/>termination check, current = initial agent"]
    resume --> lim
```

The coordinator reads handoffs from the agents' replies: the tools come from `handoff_tool_spec`, which you attach to each agent yourself, and they are declaration-only, so the agent's tool loop returns the call instead of running it. A handoff to a name that does not resolve, or that `add_handoff` does not allow, is answered with an error tool result and the same agent runs again. In the default human-in-the-loop mode the conversation is kept in shared state (so it is checkpointed) while the run waits, and the user's answer goes to the initial agent. See [handoff](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/#handoff).

### Magentic

```mermaid
flowchart TD
    inp(["task"]) --> plan["task ledger: facts + plan (two model calls)"]
    plan --> review{"with_plan_review?"}
    review -- no --> rounds
    review -- yes --> rq["request_info(MagenticPlanReviewRequest)"]
    rq -. "Revise: apply edit or replan,<br/>ask again (up to max_plan_review_rounds)" .-> rq
    rq -. "Approve: apply edit or replan from comments" .-> rounds
    rounds{"round or reset limit hit?"} -- yes --> partial["yield last assistant message as partial result"]
    rounds -- no --> ledger["progress ledger (retried if unparseable)"]
    ledger -- "unparseable after retries" --> reset
    ledger --> sat{"request satisfied?"}
    sat -- yes --> final["final answer"] --> out(["Output: final answer"])
    sat -- no --> stall["stall count +1 if no progress or in a loop,<br/>otherwise −1"]
    stall --> over{"stall count over max_stall_count?"}
    over -- "yes, with_stall_intervention" --> si["request_info(MagenticStallInterventionRequest)"]
    si -. "Continue: stall count = 0" .-> rounds
    si -. "Replan (optional guidance)" .-> reset
    si -. "Abort" .-> final
    over -- yes --> reset["reset history, replan"] --> rounds
    over -- no --> speak{"next speaker known?"}
    speak -- no --> final
    speak -- yes --> agent["append instruction, run participant over the history"] --> rounds
```

The manager plans first, then each round fills in a progress ledger that says whether the task is done, whether the team is stuck, and who speaks next with what instruction. Stalls accumulate and productive rounds pay them back; past `max_stall_count` the manager resets and replans, or, with stall intervention, asks a person, who can continue, force a replan with guidance, or stop with a final answer. Plan review works the same way before the first round: `Approve` proceeds (adopting an edited plan, or replanning from comments), `Revise` asks again until `max_plan_review_rounds`. Without `max_round_count` or `max_reset_count` the only bound is that cycle, so set a round limit. See [Magentic](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/#magentic).

### Agent approval

```mermaid
sequenceDiagram
    participant Prev as previous node
    participant AX as AgentApprovalExecutor
    participant Agent
    participant App

    Prev->>AX: conversation
    AX->>Agent: run(conversation)
    Agent-->>AX: reply
    AX-->>App: RequestInfo(ApprovalRequest { conversation, reply })
    alt empty answer: null, blank string, [] or {}
        App->>AX: send_response(null)
        AX->>AX: append reply, send on
    else feedback
        App->>AX: send_response("shorter, please")
        AX->>Agent: run(conversation + feedback as user message)
        Agent-->>AX: new reply
        AX-->>App: RequestInfo again
    end
```

`AgentApprovalExecutor` pauses after the agent has replied, so a person reviews the actual reply. An empty answer approves; anything else is feedback, the agent runs again and a new request is raised. The executor keeps no state of its own: the request carries the conversation and reply, so a checkpoint taken while it waits needs nothing else. See [agent approval](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/#agent-approval).

## Hosting

```mermaid
flowchart LR
    subgraph clients["clients"]
        dev["DevUI-style client / curl"]
        oai["OpenAI SDK"]
        a2ac["A2A client (A2AAgent)"]
        ag["AG-UI frontend (CopilotKit) / AgUiChatClient"]
    end
    subgraph app["your axum app"]
        sec["HostingSecurity (optional)<br/>bearer token, allowed hosts"]
        host["AgentHost::into_router<br/>GET /health, /v1/entities, /v1/entities/{id}/info<br/>POST /v1/responses, GET / and /ui"]
        compat["OpenAiRouter<br/>POST /v1/chat/completions"]
        a2a["A2ARouter<br/>GET /.well-known/agent-card.json<br/>POST / (JSON-RPC)"]
        agui["AgUiRouter<br/>POST {path} (SSE)"]
    end
    dev --> sec --> host
    oai --> sec --> compat
    a2ac --> sec --> a2a
    ag --> sec --> agui
    host --> ent["agents and workflows<br/>(SupportsAgentRun, Workflow)"]
    compat --> ent
    a2a --> ent
    agui --> ent
```

Each surface is an `axum::Router` you merge or nest into one app. All of them call the agent through `SupportsAgentRun` (`run`, or `run_stream` for streamed requests), and none keeps a conversation between requests, except AG-UI with a snapshot store. A2A `message/send` completes each task synchronously. `AgentHost::with_bearer_token` and `with_allowed_hosts` protect only the DevUI routes, so wrap the composed router with `HostingSecurity` to protect every surface. See [securing and composing](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_hosting/#securing-and-composing).

## MCP client lifecycle

```mermaid
sequenceDiagram
    participant Agent
    participant Src as McpStdioTool (ToolSource)
    participant Client as McpClient
    participant Server as MCP server

    Agent->>Src: resolve_tools() (every run)
    alt first use
        Src->>Server: spawn child (stdio) or connect (HTTP, WebSocket)
        Src->>Client: initialize(): protocol 2025-06-18, sampling and roots if configured
        Client->>Server: initialize + notifications/initialized
    end
    Src->>Client: list_tools_cached()
    alt cache empty or invalidated
        Client->>Server: tools/list (follows cursors)
    end
    Client-->>Src: descriptors
    Src-->>Agent: ToolDefinitions (allowed_tools filter, approval mode)
    Agent->>Src: model calls a tool → executor
    Src->>Server: tools/call
    Server-->>Src: CallToolResult
    Server--)Client: notifications/tools/list_changed → cache invalidated
    Server->>Client: sampling/createMessage → SamplingHandler → ChatClient
    Agent->>Src: close() (final: later calls fail)
```

A tool type owns one lazily created session. Registered with `tool_source`, it is asked for its tools at the start of every run: the first call spawns or connects and runs the handshake, and later calls are served from a cache that the server's `list_changed` notification clears. `tool_definitions()` is the one-off alternative that freezes the list at build time. A failure to connect or list fails the agent run. There is no reconnection: once the server dies or `close()` is called, build a new tool. See [lifecycle, errors and reconnecting](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_mcp/#lifecycle-errors-and-reconnecting).

## Telemetry spans

```mermaid
flowchart TD
    ia["invoke_agent {agent}<br/>(Agent::run, run_stream with agent middleware)"]
    ia --> c1["chat {model}<br/>(ObservableChatClient)"]
    ia --> t1["execute_tool {tool}<br/>(tool loop)"]
    ia --> c2["chat {model}"]
```

One run with one tool call produces this tree when the provider client is wrapped in `ObservableChatClient`: the agent span from the run, one `chat` span per model request from the decorator, and an `execute_tool` span per local call from the tool loop. Where `ObservableChatClient` sits relative to `RetryingChatClient` decides whether a retried request is one span or one per attempt. See [observability](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_observability/#spans).

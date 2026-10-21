---
title: "Mistral Agents API: Diagrams"
description: "Where things run, agent versions, conversation flow, function calls, approvals, handoffs and streaming events in Mistral's Agents and Conversations APIs."
framework: "mistral-agents-api"
language: "python"
sidebar:
  label: "Diagrams"
  order: 9
---

Diagrams of the Agents and Conversations APIs as a client sees them. Endpoints, entry types and event names match `mistralai` %%version=mistralai%% (checked %%versions-checked%%). They show the request and response contract, not Mistral's internal systems. Code for each step is in the [core reference](/mistral-agents-api-guide/mistral_agents_api_comprehensive_guide/).

## Where things run

```
YOUR PROCESS                                  MISTRAL
────────────                                  ───────
Mistral client ───── HTTPS ─────────────────► /v1/agents          stored, versioned agents
                                              /v1/conversations   stored history, agent loop
your function tools ◄── function.call ──────  model
   └── FunctionResultEntry ─── append() ────►
                                              built-in tools:     web_search(_premium),
                                                                  code_interpreter,
                                                                  image_generation,
                                                                  document_library
                                              registered connectors ──► remote MCP servers
client-side MCP servers (RunContext) ◄─ stdio / HTTP, called by your process
```

## Agent versions and aliases

```
create()                 POST  /v1/agents                      → version 0
update(...)              PATCH /v1/agents/{id}                 → new version (1, 2, …), becomes current
update_version(version)  PATCH /v1/agents/{id}/version         → make an older version current (rollback)
list_versions()          GET   /v1/agents/{id}/versions
create_version_alias()   PUT   /v1/agents/{id}/aliases         → alias "prod" → version N
delete()                 DELETE /v1/agents/{id}

           v0 ──update──► v1 ──update──► v2   (current)
                           ▲
                   alias "prod"

conversations.start(agent_id=..., agent_version="prod")  → runs v1
conversations.start(agent_id=...)                        → runs the current version
```

## Conversation lifecycle

```
start()     POST /v1/conversations                       → conversation_id, outputs[]
append()    POST /v1/conversations/{id}                  → outputs[] for the new turn
restart()   POST /v1/conversations/{id}/restart          → NEW conversation sharing history
                                                           up to from_entry_id
get_history()   GET /v1/conversations/{id}/history       → every entry
get_messages()  GET /v1/conversations/{id}/messages      → messages only
delete()        DELETE /v1/conversations/{id}

conv_A:  [in1] [out1] [in2] [out2]
                  │
        restart(from_entry_id=out1, inputs="…")
                  ▼
conv_B:  [in1] [out1] [in3] [out3]        conv_A is unchanged
```

`store=False` on `start()` runs a single request without keeping it; there is nothing to append to afterwards.

## Entries in `outputs`

```
type               model class            produced when
────               ───────────            ─────────────
message.input      MessageInputEntry      your input (in history)
message.output     MessageOutputEntry     the model answers; content = str or chunks
                                          (TextChunk, ToolReferenceChunk, ToolFileChunk)
tool.execution     ToolExecutionEntry     a built-in tool or connector ran on Mistral's side
function.call      FunctionCallEntry      the model wants one of YOUR functions
                                          (or a tool awaiting approval: confirmation_status="pending")
function.result    FunctionResultEntry    you send this back with append()
agent.handoff      AgentHandoffEntry      the conversation moved to another agent
```

A typical `start()` with web search returns `[tool.execution, message.output]`; with a function tool it can end at `[function.call]`, waiting for you.

## Function call loop

```
your code                                   Mistral
─────────                                   ───────
start(agent_id, inputs) ──────────────────► model decides to call get_order_status
          ◄──────────── outputs[-1] = function.call(tool_call_id="tc1", arguments)
run get_order_status(**arguments)
append(conversation_id,
       inputs=[FunctionResultEntry(tool_call_id="tc1", result=…)]) ──►
          ◄──────────── outputs[-1] = message.output   (or another function.call: loop)
```

`RunContext` with `run_async()` runs the same loop for you, and also calls client-side MCP servers.

## Approvals

```
Connector or built-in tool with tool_configuration.requires_confirmation:

start(...) ───────────────────────────────► model calls post_comment
          ◄── function.call(confirmation_status="pending", tool_call_id="tc7")
(ask a person; persist conversation_id + tc7 if this takes a while)
append(conversation_id,
       tool_confirmations=[{"tool_call_id": "tc7", "confirmation": "allow" | "deny"}]) ──►
          ◄── the tool runs, then message.output (allow)   or   message.output only (deny)

Your function with RunContext.register_func(fn, requires_confirmation=True):

run_async() ──► raises DeferredToolCallsException
                  │ to_dict() → store → from_dict()
                  ▼
run_async(inputs=[call.confirm() | call.reject("reason"), …])  with the same conversation_id
```

## Handoffs

```
handoff_execution="server" (default)        handoff_execution="client"
────────────────────────────────────        ──────────────────────────
start(agent_id=triage)                      start(agent_id=triage)
  triage ── agent.handoff ──► billing         triage ── agent.handoff ──► (stop)
  billing answers in the same request       ◄── outputs[-1] = agent.handoff
◄── [agent.handoff, message.output]         your code checks, logs or blocks,
append() continues with billing             then runs next_agent_id itself
```

## Streaming events

`start_stream()`, `append_stream()` and `restart_stream()` send the same request with `stream: true` and return server-sent events:

```
conversation.response.started      conversation_id
tool.execution.started             name, arguments        ┐ zero or more,
tool.execution.delta / .done                              ┘ per tool
function.call.delta                                         your function's arguments
agent.handoff.started / .done                               on a handoff
message.output.delta               content (repeated)       the answer, piece by piece
conversation.response.done         usage                    last event
conversation.response.error        message, code            instead of .done on failure
```

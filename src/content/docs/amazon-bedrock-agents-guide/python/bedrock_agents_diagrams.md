---
title: "Amazon Bedrock Agents: Architecture Diagrams"
description: "Diagrams of AgentCore Runtime, the managed harness, Memory, Gateway and Identity, Strands multi-agent patterns, and Bedrock Agents Classic orchestration, action groups and sessions."
framework: "amazon-bedrock-agents"
language: "python"
sidebar:
  label: "Diagrams"
  order: 11
---

How the pieces fit together. Each diagram links to the page that covers it in detail.

## AgentCore

### Runtime request flow

One `runtimeSessionId` maps to one microVM. Reusing the ID returns to the same microVM and its in-process state until the idle timeout or maximum lifetime ends the session ([AgentCore guide](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/#secure-sandbox-runtime)).

```text
 caller (app, other agent, MCP/A2A client)
   │  InvokeAgentRuntime(arn, qualifier, runtimeSessionId, payload)
   │  auth: SigV4 (IAM)  or  Bearer JWT (customJWTAuthorizer)
   ▼
┌──────────────────────────────────────────────────────────────┐
│ AgentCore Runtime                                            │
│   endpoint "prod" ──► v4        endpoint DEFAULT ──► latest  │
│                                                              │
│   ┌─ session microVM (one per runtimeSessionId) ───────────┐ │
│   │  your code: BedrockAgentCoreApp on port 8080           │ │
│   │    GET /ping          Healthy or HealthyBusy           │ │
│   │    POST /invocations  @app.entrypoint(payload, context)│ │
│   │  or an MCP server (/mcp) or A2A server (port 9000)     │ │
│   └──────────────┬──────────────┬──────────────┬───────────┘ │
└──────────────────┼──────────────┼──────────────┼─────────────┘
                   ▼              ▼              ▼
             Bedrock models   AgentCore      Gateway, Code
             (InvokeModel)    Memory         Interpreter, Browser
                   │
                   └── spans, logs, metrics ──► CloudWatch (Observability)
```

### Harness or code-defined agent

```text
                    ┌──────────────── AgentCore Runtime ─────────────────┐
 InvokeHarness ───► │ managed harness                                    │
  (messages,        │   harness.json: model, system prompt, tools,       │
   per-call         │   skills, memory, limits, lifecycle hooks          │
   overrides)       │   AWS runs the agent loop                          │
                    │                                                    │
 InvokeAgent   ───► │ code-defined agent                                 │
  Runtime           │   your container or zip: Strands, LangGraph,       │
  (payload)         │   CrewAI, OpenAI Agents SDK... you own the loop    │
                    └────────────────────────────────────────────────────┘

   agentcore create --defaults        ──► harness project
   agentcore create --framework ...   ──► code project
   agentcore export harness           ──► harness converted to Strands code
   agentcore create --type import     ──► Agents Classic agent converted to code
```

See [Managed harness](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/#managed-harness).

### Memory

Short-term memory is the raw event log. Long-term records are extracted from it asynchronously by the strategies you enable, and stored under namespaces scoped by actor (and optionally session).

```text
 agent turn
   │ create_event(memory_id, actor_id, session_id, messages)
   ▼
┌──────────────────────────┐   async extraction   ┌─────────────────────────────┐
│ short-term: events       │ ───────────────────► │ long-term: memory records   │
│ per actor + session      │   strategies:        │ /users/{actorId}/facts      │
│ expire after N days      │   SEMANTIC           │ /users/{actorId}/...        │
└──────────────────────────┘   USER_PREFERENCE    │ /summaries/{actorId}/{sid}  │
   ▲  list_events             SUMMARIZATION       └──────────────┬──────────────┘
   │  (recent turns)          EPISODIC                           │
   │                                       retrieve_memories(namespace, query)
   └──────────── agent context for next turn ◄───────────────────┘
```

With Strands, `AgentCoreMemorySessionManager` makes these calls for you ([Strands guide](/strands-agents-guide/python/strands_comprehensive_guide/#agentcore-memory)).

### Gateway and Identity

```text
                 inbound auth (JWT or IAM)
 agent ──── MCP over HTTP ────► ┌──────────────────────────┐
 (MCPClient(url=gatewayUrl))    │ AgentCore Gateway        │
                                │  tools/list, tools/call  │
                                │  semantic tool search    │
                                │  optional Cedar policies │
                                └───┬─────────┬─────────┬──┘
                    outbound creds: │IAM role │ OAuth / │ API key
                                    ▼         ▼         ▼
                              Lambda fn   OpenAPI /   MCP server,
                                          Smithy API  knowledge base

 AgentCore Identity
   inbound:  user JWT ──► Runtime/Gateway authorizer ──► workload access token
   outbound: workload token ──► token vault ──► OAuth token or API key for a tool
             (@requires_access_token, @requires_api_key)
```

See [Integration patterns](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/#integration-patterns) and [Identity](/amazon-bedrock-agents-guide/python/bedrock_agents_production_guide/#identity).

### Deployment with the CLI

```text
 agentcore/agentcore.json ─┐
 agentcore/aws-targets.json├─► agentcore deploy --target prod
 app/<Agent>/ (code)       ┘        │
                                    ▼
                         CDK app (agentcore/cdk) ──► CloudFormation stack
                                    │
          ┌─────────────┬───────────┼────────────┬──────────────┐
          ▼             ▼           ▼            ▼              ▼
     IAM execution   Runtime +   Memory     Gateway +     Online eval
     role            endpoints              targets       configs
     (CodeZip in S3, or ECR image built by CodeBuild)
```

See the [production guide](/amazon-bedrock-agents-guide/python/bedrock_agents_production_guide/).

## Strands multi-agent patterns

```text
 agents as tools              swarm                        graph (GraphBuilder)
 ───────────────              ─────                        ────────────────────
     supervisor           ┌──────────┐ handoff ┌───────┐   collect ─► validate ─┬─► analyse ─► report
     │       │            │researcher│────────►│analyst│      ▲                 │
     ▼       ▼            └──────────┘         └───┬───┘      └── FAIL (bounded)┘
  sales   billing              ▲     handoff       │
  .as_tool() .as_tool()        └───────────── reviewer ──► answer
  model decides           one agent active at a time;   edges with conditions; ready
  which to call           shared context, handoff tool  nodes run in parallel
```

See the [Strands guide](/strands-agents-guide/python/strands_multi_agent/#agents-as-tools) and [multi-agent patterns](/amazon-bedrock-agents-guide/python/bedrock_agents_advanced_multi_agent_python/).

## Bedrock Agents Classic

These cover Agents Classic, which has been in maintenance mode since 30 July 2026.

### Orchestration and action groups

```text
 invoke_agent(agentId, agentAliasId, sessionId, inputText, sessionState)
   │
   ▼
┌─ Bedrock Agents (managed) ─────────────────────────────────────────┐
│ pre-processing (optional) ─► orchestration loop ─► post-processing │
│                               │  model reasons, picks an action    │
│              ┌────────────────┼──────────────────┐                 │
│              ▼                ▼                  ▼                 │
│     action group:      action group:      knowledge base           │
│     Lambda executor    RETURN_CONTROL     Retrieve (vector store)  │
│        │                     │                                     │
│        ▼                     │  guardrail on input and output      │
│   your Lambda fn             │                                     │
│   (messageVersion 1.0        │                                     │
│    response envelope)        │                                     │
└──────────────────────────────┼─────────────────────────────────────┘
                               ▼
              completion stream: returnControl event
              your app runs the call, then invoke_agent again with
              sessionState.returnControlInvocationResults
```

### Sessions and memory

Agents Classic keeps the conversation **inside the service**. You don't provide a database for it.

```text
 sessionId ──► session context held by Bedrock Agents
                 kept until idleSessionTTLInSeconds passes with no calls
                 sessionAttributes        (passed to action-group Lambdas)
                 promptSessionAttributes  (added to the prompt)

 memoryConfiguration (optional, SESSION_SUMMARY)
   at session end ──► summary stored per memoryId, kept storageDays
   next session with the same memoryId ──► summaries added to context
```

See the [Classic reference](/amazon-bedrock-agents-guide/python/bedrock_agents_comprehensive_guide/#sessions-and-memory).

### Multi-agent collaboration

```text
                 ┌─────────────────────────────┐
 user ─────────► │ supervisor agent            │
                 │ agentCollaboration:         │
                 │   SUPERVISOR (plans)        │
                 │   SUPERVISOR_ROUTER (routes)│
                 └───┬─────────────┬───────────┘
                     │ associate_agent_collaborator (alias ARN, instruction)
                     ▼             ▼
               collaborator   collaborator
               agent alias    agent alias
               (own action groups and knowledge bases)
```

See [multi-agent collaboration](/amazon-bedrock-agents-guide/python/bedrock_agents_advanced_multi_agent_python/#4-multi-agent-collaboration-ga-march-2025).

---
title: "Claude Agent SDK - Architecture & Flow Diagrams"
description: "Architecture, agent loop, permission, session and deployment diagrams for the Python Claude Agent SDK."
framework: anthropic-claude-agent-sdk
---

# Claude Agent SDK - Architecture & Flow Diagrams

> Architecture and flow diagrams. Verified against `claude-agent-sdk` %%version=claude-agent-sdk%% (checked %%versions-checked%%).

---

## Table of Contents

1. [Core Architecture](#core-architecture)
2. [Agent Lifecycle](#agent-lifecycle)
3. [Tool Execution Pipeline](#tool-execution-pipeline)
4. [Multi-Agent Orchestration](#multi-agent-orchestration)
5. [MCP Integration](#mcp-integration)
6. [Session Management](#session-management)
7. [Permission System](#permission-system)
8. [Error Handling & Recovery](#error-handling--recovery)
9. [Context Compaction](#context-compaction)
10. [Deployment Architecture](#deployment-architecture)

---

## Core Architecture

### High-Level System Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                    Your application                                │
│  (API server, worker, scheduled job, CLI tool)                     │
└────────────────────┬───────────────────────────────────────────────┘
                     │ query() / ClaudeSDKClient
        ┌────────────▼─────────────────┐
        │   claude-agent-sdk (Python)  │
        │   options, typed messages,   │
        │   in-process MCP tools,      │
        │   hook + can_use_tool calls  │
        └────────────┬─────────────────┘
                     │ JSON stream over stdin/stdout
        ┌────────────▼──────────────────────────────────────────┐
        │   Claude Code CLI (subprocess, bundled in the wheel)  │
        │   agent loop · built-in tools · permissions ·         │
        │   compaction · sessions on disk · external MCP        │
        └────────────┬──────────────────────────────────────────┘
                     │ HTTPS
          ┌──────────▼─────────────────────┐
          │   Claude API / Bedrock / Vertex│
          └──────────┬─────────────────────┘
                     │
        ┌────────────▼───────────────────────┐
        │  Claude models                     │
        │  • claude-opus-5-5   (default)     │
        │  • claude-sonnet-5-5 (balanced)    │
        │  • claude-haiku-5-5  (cheapest)    │
        │  • claude-fable-5-1  (most capable)│
        └────────────────────────────────────┘
```

### Component Responsibilities

```
┌─────────────────────────────────────────────────────────┐
│               Agent Execution Engine                    │
├─────────────────────────────────────────────────────────┤
│ • Manages agent lifecycle                              │
│ • Orchestrates message flow                            │
│ • Handles streaming & buffering                        │
│ • Coordinates tool execution                           │
│ • Manages session state                                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│            Built-in Tool Ecosystem                      │
├─────────────────────────────────────────────────────────┤
│ • File Operations (Read, Write, Edit)                  │
│ • Command Execution (Bash)                             │
│ • Search Operations (Glob, Grep)                       │
│ • Web (WebSearch, WebFetch)                            │
│ • Subagents (Agent tool), TodoWrite, skills            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│      Model Context Protocol (MCP) Support               │
├─────────────────────────────────────────────────────────┤
│ • Custom Tool Servers                                  │
│ • External Service Integration                         │
│ • Resource Management                                  │
│ • Standardised Tool Interface                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│        Context & Token Management                       │
├─────────────────────────────────────────────────────────┤
│ • Automatic Context Compaction                         │
│ • Token Usage Tracking                                 │
│ • Context Window Optimisation                          │
│ • Summarisation & Pruning                              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│        Session Persistence & Recovery                   │
├─────────────────────────────────────────────────────────┤
│ • Session State Storage                                │
│ • Multi-turn Conversation Management                   │
│ • Session Forking for Branching                        │
│ • State Isolation Between Sessions                     │
└─────────────────────────────────────────────────────────┘
```

---

## Agent Lifecycle

### Complete Agent Execution Flow

```
START
  │
  ├─ 1. INITIALIZATION
  │   ├─ Load configuration
  │   ├─ Authenticate API key
  │   ├─ Create session
  │   └─ Initialize context
  │
  ├─ 2. PROMPT PROCESSING
  │   ├─ Accept user prompt
  │   ├─ Validate input
  │   ├─ Inject system prompt
  │   └─ Prepare context
  │
  ├─ 3. API CALL
  │   ├─ Send to Claude
  │   ├─ Wait for response
  │   └─ Stream or buffer
  │
  ├─ 4. REASONING & PLANNING
  │   ├─ Claude thinks about task
  │   ├─ Plans approach
  │   └─ Determines needed tools
  │
  ├─ 5. TOOL EXECUTION (Loop)
  │   │
  │   ├─ Does Claude want tools?
  │   │  ├─ YES → 6. TOOL USE
  │   │  └─ NO  → 7. FINAL RESPONSE
  │   │
  │   ├─ 6. TOOL USE
  │   │   ├─ Check permissions
  │   │   ├─ Validate input
  │   │   ├─ Execute tool
  │   │   ├─ Capture output
  │   │   └─ Return to Claude
  │   │
  │   └─ Loop back to 5 until done
  │
  ├─ 7. FINAL RESPONSE
  │   ├─ Claude generates final answer
  │   ├─ Format output
  │   └─ Send to client
  │
  ├─ 8. COMPLETION
  │   ├─ Calculate token usage
  │   ├─ Update session state
  │   ├─ Log metrics
  │   └─ Release resources
  │
  └─ END

ALTERNATIVE PATHS:
  
  ERROR HANDLING:
    • Validation fails → Reject input
    • Permission denied → Deny tool use
    • Tool fails → Retry or alternative
    • API error → Exponential backoff
    • Rate limit → Queue or wait
    • Timeout → Graceful shutdown
    • Budget exceeded → Halt execution

  SESSION CONTINUATION:
    • Resume existing session
    • Retrieve conversation history
    • Maintain context
    • Continue from checkpoint
```

### Single-Turn vs Multi-Turn

```
SINGLE-TURN AGENT:
  User Input → [Agent] → Response → END

MULTI-TURN AGENT:
  Session Created
       │
       ├─ Turn 1: Input → [Agent] → Response
       ├─ Turn 2: Input → [Agent] → Response
       ├─ Turn 3: Input → [Agent] → Response
       └─ Turn N: Input → [Agent] → Response
       │
  Session Closed / Resumed

FORKED SESSION:
  Original Session
       │
       ├─ Main Branch: Continue normally
       └─ Fork: Branch off for alternative approach
         (Both maintain separate context)
```

---

## Tool Execution Pipeline

### Tool Invocation Flow

```
Claude wants to use a tool
  │
  ▼
Tool Request Generated
  ├─ Tool Name
  ├─ Parameters
  └─ Tool ID
  │
  ▼
PERMISSION CHECK
  ├─ Policy evaluation
  ├─ Pattern matching
  └─ User approval?
  │
  ├─ DENIED ──→ Reject to Claude
  │             (Claude will try alternative)
  │
  ├─ ALLOW ───→ Continue
  │
  └─ ASK ─────→ Prompt user for approval
              │
              ├─ Approved? → Continue
              └─ Denied?   → Reject to Claude
  │
  ▼
INPUT VALIDATION
  ├─ Type checking
  ├─ Format validation
  ├─ Sanitisation
  └─ Security checks
  │
  ├─ INVALID ──→ Error message to Claude
  │               (Claude retries)
  │
  └─ VALID ────→ Continue
  │
  ▼
TOOL EXECUTION
  ├─ File ops: Read/Write/Edit
  ├─ Commands: Bash/Terminal
  ├─ Search: Glob/Grep/Web
  ├─ Custom: MCP tools
  ├─ Subagents: Agent tool
  └─ Custom hooks (Pre/Post)
  │
  ├─ SUCCESS ──→ Capture output
  │
  ├─ FAILURE ──→ Error information
  │
  └─ TIMEOUT ──→ Abort & error
  │
  ▼
RESULT FORMATTING
  ├─ Sanitise sensitive data
  ├─ Truncate if too large
  ├─ Format for Claude
  └─ Add context
  │
  ▼
RETURN TO CLAUDE
  ├─ Tool result message
  ├─ Any errors or notes
  └─ Request next action
  │
  ▼
Claude processes result
  └─ Uses info to continue task
```

### Built-in Tools Overview

```
FILE OPERATIONS:
  Read          │ Write         │ Edit          │ Glob
  ├─ Read file  │ ├─ Create     │ ├─ Modify     │ ├─ List files
  ├─ Get content│ ├─ Overwrite  │ ├─ Insert     │ └─ Pattern match
  └─ Return text│ └─ Save       │ └─ Delete line│

SEARCH OPERATIONS:
  Glob                    │ Grep
  ├─ Recursive search     │ ├─ Pattern matching
  ├─ Wildcard patterns    │ ├─ Regex support
  └─ File list return     │ └─ Context lines

COMMAND EXECUTION:
  Bash / Shell
  ├─ Execute commands
  ├─ Capture stdout
  ├─ Return stderr
  └─ Exit code

WEB:
  WebSearch              │ WebFetch
  ├─ Query the web       │ ├─ Fetch a URL
  └─ Return results      │ └─ Return page content

DELEGATION:
  Agent
  ├─ Start a subagent defined in options.agents
  ├─ Own context window, tools and model
  └─ Returns a summary to the parent

---

## Multi-Agent Orchestration

### Sequential Agent Pattern

```
Task
 │
 ├─ Agent 1: Analysis
 │   ├─ Process data
 │   ├─ Identify patterns
 │   └─ Output: Findings
 │
 ├─ Agent 2: Planning (uses findings from Agent 1)
 │   ├─ Receive findings
 │   ├─ Create plan
 │   └─ Output: Action plan
 │
 ├─ Agent 3: Implementation (uses plan from Agent 2)
 │   ├─ Receive plan
 │   ├─ Execute plan
 │   └─ Output: Results
 │
 └─ Agent 4: Review (uses results from Agent 3)
     ├─ Receive results
     ├─ Quality check
     └─ Output: Final report

Result
```

### Parallel Agent Pattern

```
                    Task
                     │
         ┌───────────┼───────────┐
         │           │           │
         ▼           ▼           ▼
    Security      Performance   Quality
    Review        Analysis      Review
    Agent 1       Agent 2       Agent 3
         │           │           │
         ├─ Find     ├─ Identify ├─ Check for
         │  vulns    │  bottle   │  issues
         │           │  necks    │
         │           │           │
         └───────────┼───────────┘
                     │
                AGGREGATE RESULTS
                     │
                Final Report
```

### Hierarchical Agent Pattern

```
                 Coordinator
                 (Main Agent)
                      │
         ┌────────────┼────────────┐
         │            │            │
      Specialist   Specialist   Specialist
      Agent 1      Agent 2      Agent 3
      (Security)   (Performance)(Quality)
         │            │            │
    ├─ Tool 1    ├─ Tool 1    ├─ Tool 1
    ├─ Tool 2    ├─ Tool 2    ├─ Tool 2
    └─ Tool 3    └─ Tool 3    └─ Tool 3
         │            │            │
    Findings      Findings      Findings
         │            │            │
         └────────────┼────────────┘
                  │
              Synthesis
                  │
            Final Decision
```

### Agent Communication Patterns

```
DIRECT COMMUNICATION:
  Agent A → Output → Agent B
  (Sequential dependency)

MESSAGE PASSING:
  Agent A ─┐
           ├─→ Shared Queue ─→ Agent B
  Agent C ─┘

BROADCAST:
  Agent A broadcasts to:
    ├─ Agent B
    ├─ Agent C
    └─ Agent D

HIERARCHICAL:
  Sub-agent 1  Sub-agent 2  Sub-agent 3
        │            │            │
        └────────────┼────────────┘
                     │
                Coordinator
                     │
                  Result
```

---

## MCP Integration

### MCP Server Architecture

```
Claude Application
      │
      ▼
MCP Client (SDK)
      │
      ├─ Server 1 (stdio)
      │   ├─ Custom Tools
      │   ├─ Business Logic
      │   └─ Resources
      │
      ├─ Server 2 (HTTP/SSE)
      │   ├─ Remote Tools
      │   ├─ External APIs
      │   └─ Shared Resources
      │
      ├─ Server 3 (SDK MCP)
      │   ├─ In-process Tools
      │   ├─ Direct Integration
      │   └─ No Subprocess
      │
      └─ Server N...
```

### Tool Registration & Discovery

```
Server Starts
      │
      ▼
Advertise Tools
      │
      ├─ Tool 1: get_weather
      │   ├─ Parameters: location, units
      │   ├─ Description: Get weather
      │   └─ Handler: async function
      │
      ├─ Tool 2: calculate
      │   ├─ Parameters: expression
      │   ├─ Description: Math calc
      │   └─ Handler: async function
      │
      └─ Tool N...
      │
      ▼
Available to Claude
      │
Claude Uses Tool
      │
      ├─ Tool call request
      ├─ Parameter validation
      ├─ Execute handler
      └─ Return result
```

### MCP Server Lifecycle

```
Start MCP Server
      │
      ▼
Initialize Tools
      │
      ├─ Load handlers
      ├─ Setup validation
      └─ Configure parameters
      │
      ▼
Listen for Requests
      ├─ Tool calls
      ├─ Resource access
      └─ Server queries
      │
  ┌───┴───┐
  │       │
  ▼       ▼
Request Processing
  ├─ Parse params
  ├─ Validate input
  ├─ Execute
  ├─ Format output
  └─ Send response
      │
  Repeat until shutdown
      │
      ▼
Graceful Shutdown
      ├─ Close resources
      ├─ Release connections
      └─ Exit
```

---

## Session Management

### Session Lifecycle

```
CREATE SESSION
      │
      ├─ Session ID generated
      ├─ Context initialised
      ├─ Storage allocated
      └─ Timestamp recorded
      │
      ▼
ACTIVE SESSION
      ├─ Turn 1
      │   ├─ User input
      │   ├─ Claude processes
      │   └─ Tool execution
      │
      ├─ Turn 2
      │   ├─ User input
      │   ├─ Claude processes
      │   └─ Tool execution
      │
      └─ Turn N...
      │
      ├─ SUSPEND SESSION (inactive)
      │   ├─ Save state
      │   ├─ Release resources
      │   └─ Retain in storage
      │
      ├─ RESUME SESSION
      │   ├─ Load saved state
      │   ├─ Restore context
      │   ├─ Continue from checkpoint
      │   └─ Allocate resources
      │
      ├─ FORK SESSION
      │   ├─ Checkpoint current
      │   ├─ Create branch
      │   ├─ Independent state
      │   └─ Separate context
      │
      ▼
CLOSE SESSION
      ├─ Archive results
      ├─ Final analytics
      ├─ Cleanup resources
      └─ Store for history
```

### Session State Persistence

```
LOCAL (default)                       EXTERNAL (session_store=...)
┌──────────────────────────┐          ┌────────────────────────────┐
│ ~/.claude/projects/<key>/│  mirror  │ SessionStore implementation│
│   <session_id>.jsonl     │ ───────▶ │ (Redis, S3, database...)   │
│ one JSON line per entry  │          │ append() per batch/entry   │
└──────────────────────────┘          └────────────────────────────┘

resume=<id>:   read local transcript, or rebuild it from the store
fork_session:  copy history into a new session ID
list_sessions / get_session_messages: read transcripts without the CLI
```

---

## Permission System

### Permission Architecture Layers

```
TOOL CALL REQUESTED
  │
  ▼
1. HOOKS (PreToolUse)
  │   may allow, deny, ask, or rewrite the input
  ▼
2. DENY RULES (disallowed_tools, settings)
  │   match → rejected, Claude is told why
  ▼
3. PERMISSION MODE
  │   bypassPermissions → run
  │   acceptEdits       → file edits run
  │   plan              → nothing executes
  │   dontAsk           → anything not pre-approved is denied
  ▼
4. ALLOW RULES (allowed_tools, settings)
  │   match → run, can_use_tool is NOT called
  ▼
5. can_use_tool CALLBACK
  │   PermissionResultAllow(updated_input=...) → run
  │   PermissionResultDeny(message=..., interrupt=...) → reject
  ▼
EXECUTION (optionally inside the Bash sandbox)
  │
  ▼
HOOKS (PostToolUse / PostToolUseFailure)
```

### Permission Modes

```
MODE: default
  ├─ Unapproved tools go to can_use_tool
  └─ Use when: production, with a callback or narrow allow rules

MODE: acceptEdits
  ├─ File edits auto-approved
  └─ Use when: trusted coding tasks in a disposable checkout

MODE: plan
  ├─ Claude plans; no edits or commands execute
  └─ Use when: you want a plan to review first

MODE: dontAsk
  ├─ Anything not pre-approved is denied
  └─ Use when: unattended jobs with a fixed tool set

MODE: auto
  ├─ The CLI's classifier decides whether a call is safe
  └─ Use when: interactive tools that want fewer prompts

MODE: bypassPermissions
  ├─ Everything runs, no checks
  └─ Use when: inside a throwaway sandbox only
```

---

## Error Handling & Recovery

### Error Detection & Response Flow

```
OPERATION
  │
  ▼
ERROR DETECTED?
  │
  ├─ NO → Continue normally
  │
  ├─ YES → Error Classification
  │   │
  │   ├─ TRANSIENT
  │   │  ├─ Rate limit
  │   │  ├─ Network timeout
  │   │  ├─ Temporary server issue
  │   │  └─ Action: Retry with backoff
  │   │
  │   ├─ AUTHENTICATION
  │   │  ├─ Invalid API key
  │   │  ├─ Expired credentials
  │   │  └─ Action: Fail immediately
  │   │
  │   ├─ VALIDATION
  │   │  ├─ Invalid input
  │   │  ├─ Schema mismatch
  │   │  └─ Action: Reject & feedback
  │   │
  │   ├─ RESOURCE
  │   │  ├─ Context length exceeded
  │   │  ├─ Token budget exceeded
  │   │  └─ Action: Fallback strategy
  │   │
  │   └─ PERMISSION
  │       ├─ Tool denied
  │       ├─ Access restricted
  │       └─ Action: Notify, try alternative
  │
  ▼
RECOVERY STRATEGY
  │
  ├─ Backoff & Retry
  │   └─ Exponential delays: 1s, 2s, 4s... (max 30s)
  │
  ├─ Fallback Option
  │   ├─ Try different model
  │   ├─ Use simpler approach
  │   └─ Degrade gracefully
  │
  ├─ Circuit Breaker
  │   ├─ Fail fast after N errors
  │   ├─ Wait before retry
  │   └─ Prevent cascading failures
  │
  ├─ Alerting
  │   ├─ Log error details
  │   ├─ Notify team
  │   └─ Track metrics
  │
  └─ User Communication
      ├─ Explain what failed
      ├─ Provide alternatives
      └─ Ask for help if needed
```

---

## Context Compaction

### Automatic Context Management

```
CONVERSATION GROWS
      │
      ├─ Message 1
      ├─ Message 2
      ├─ Message 3
      ├─ ...
      └─ Message 100
            │
            ▼
      Context Approaching Limit?
            │
            ├─ NO → Continue
            │
            └─ YES → Initiate Compaction
                  │
                  ▼
            SELECT COMPACTION STRATEGY
                  │
                  ├─ SUMMARISATION
                  │   ├─ Keep recent 5 messages
                  │   ├─ Summarise older 95
                  │   └─ Result: "Earlier discussion..."
                  │
                  ├─ PRUNING
                  │   ├─ Remove less relevant
                  │   ├─ Keep important context
                  │   └─ Maintain token count
                  │
                  └─ COMPRESSION
                      ├─ Extract key facts
                      ├─ Combine similar items
                      └─ Remove redundancy
                  │
                  ▼
            COMPACTED CONTEXT
            (50% of original size)
                  │
                  ▼
            RESUME CONVERSATION
            (With compacted history)
```

### Token Budget Tracking

```
EACH TURN
      │
      ▼
CONTEXT USAGE  (client.get_context_usage())
      │
      ├─ system prompt + tools
      ├─ memory files (CLAUDE.md)
      ├─ conversation history
      └─ tool results
            │
            ▼
      totalTokens vs maxTokens (1M on the Claude 5 family)
            │
            ├─ below autoCompactThreshold → continue
            │
            └─ above it → automatic compaction
                  └─ PreCompact hook fires first

SEPARATE LIMITS (per run)
      ├─ max_turns       → ResultMessage subtype "error_max_turns"
      ├─ max_budget_usd  → "error_max_budget_usd"
      └─ task_budget     → advisory: the model paces itself
```

---

## Deployment Architecture

### Container-Based Deployment

```
Development Machine
        │
        ▼
  Docker Image
  ├─ SDK installed
  ├─ Dependencies
  ├─ Configuration
  └─ Application code
        │
        ▼
  Container Registry
  (Docker Hub, ACR, GCR)
        │
        ▼
  Production Environment
        │
        ├─ Container 1
        ├─ Container 2
        ├─ Container 3
        └─ Container N
        │
        ├─ Load Balancer
        ├─ Configuration Service
        ├─ Logging Aggregation
        ├─ Metrics Collection
        └─ Monitoring & Alerts
```

### Kubernetes Orchestration

```
┌─────────────────────────────────────────────────┐
│         Kubernetes Cluster                      │
├─────────────────────────────────────────────────┤
│                                                 │
│  Deployment: claude-agent                       │
│  ├─ Replicas: 3                                │
│  ├─ Rolling Update Strategy                    │
│  └─ Resource Requests/Limits                   │
│                                                 │
│  Service: claude-agent                          │
│  ├─ Load Balancer                              │
│  ├─ Port Mapping                               │
│  └─ DNS Resolution                             │
│                                                 │
│  ConfigMap: Settings                            │
│  ├─ Model selection                            │
│  ├─ Logging level                              │
│  └─ Timeout values                             │
│                                                 │
│  Secret: Credentials                            │
│  ├─ API Keys                                   │
│  ├─ Database password                          │
│  └─ Encryption keys                            │
│                                                 │
│  PersistentVolumeClaim: Sessions               │
│  ├─ Session storage                            │
│  ├─ Conversation history                       │
│  └─ Metrics data                               │
│                                                 │
│  HorizontalPodAutoscaler                       │
│  ├─ Min replicas: 2                            │
│  ├─ Max replicas: 10                           │
│  ├─ Target CPU: 70%                            │
│  └─ Target Memory: 80%                         │
│                                                 │
│  NetworkPolicy                                  │
│  ├─ Ingress rules                              │
│  ├─ Egress rules                               │
│  └─ Service discovery                          │
│                                                 │
└─────────────────────────────────────────────────┘
```

### High-Availability Setup

```
USER REQUESTS
      │
      ▼
┌─────────────────────┐
│   Load Balancer     │
│ (Health checks)     │
└────────┬────────────┘
         │
    ┌────┼────┐
    │    │    │
    ▼    ▼    ▼
  [Pod] [Pod] [Pod]
   │     │     │
   └─────┼─────┘
         │
   ┌─────▼────────┐
   │ Cache Layer  │
   │ (Redis)      │
   └──────────────┘
         │
   ┌─────▼──────────┐
   │ Database       │
   │ (With Replica) │
   └────────────────┘

If one pod fails:
  └─ Load balancer redirects to healthy pods
  └─ Stateful data persists in database
  └─ New pod spins up automatically
```

---

**These diagrams provide visual references for understanding the Claude Agent SDK architecture and execution patterns. Refer back to them when designing your agent systems.**


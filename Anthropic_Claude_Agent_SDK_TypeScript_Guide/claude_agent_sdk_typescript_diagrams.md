# Claude Agent SDK (TypeScript) - Architecture Diagrams and Flows

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Claude Agent SDK Architecture                         │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────┐
    │                     Application Layer                            │
    │  (Your TypeScript Code, React Frontend, Express Servers, etc.)  │
    └────────────────────┬─────────────────────────────────────────────┘
                         │
    ┌────────────────────▼─────────────────────────────────────────────┐
    │              Claude Agent SDK Interface                          │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │ query() - Main async generator function                  │   │
    │  │ createSdkMcpServer() - MCP server creation              │   │
    │  │ tool() - Tool definition with Zod schemas              │   │
    │  └──────────────────────────────────────────────────────────┘   │
    └────────────────────┬─────────────────────────────────────────────┘
                         │
    ┌────────────────────▼─────────────────────────────────────────────┐
    │             Core Agent Harness (Built on Claude Code)           │
    │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
    │  │ Agent Loop   │  │ Context Mgmt │  │ Permission System  │    │
    │  │ Executor     │  │ Compaction   │  │ Access Control     │    │
    │  └──────────────┘  └──────────────┘  └────────────────────┘    │
    └────────────────────┬─────────────────────────────────────────────┘
                         │
    ┌────────────────────▼─────────────────────────────────────────────┐
    │                  Tool Execution Framework                        │
    │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
    │  │ Built-In     │  │ Custom MCP   │  │ Computer Use API   │    │
    │  │ Tools        │  │ Tools        │  │ (Mouse, Keyboard)  │    │
    │  │ (File, Bash) │  │              │  │                    │    │
    │  └──────────────┘  └──────────────┘  └────────────────────┘    │
    └────────────────────┬─────────────────────────────────────────────┘
                         │
    ┌────────────────────▼─────────────────────────────────────────────┐
    │                   External Services                              │
    │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
    │  │ Claude API   │  │ AWS Bedrock  │  │ Google Vertex AI   │    │
    │  │ (Direct)     │  │ (AWS Auth)   │  │ (GCP Auth)         │    │
    │  └──────────────┘  └──────────────┘  └────────────────────┘    │
    └─────────────────────────────────────────────────────────────────┘
```

## Query Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│              Query Execution Flow Diagram                        │
└─────────────────────────────────────────────────────────────────┘

1. INITIALISATION
   ┌────────┐
   │ query()│
   └────┬───┘
        │
        ▼
   ┌─────────────────────────────────┐
   │ Create Agent Instance            │
   │ - Session ID                      │
   │ - Model Selection                 │
   │ - Permission Mode                 │
   │ - Tool Registry                   │
   └────────────┬──────────────────────┘
                │
                ▼
   ┌─────────────────────────────────┐
   │ Initialise Context              │
   │ - Load system prompt            │
   │ - Set working directory         │
   │ - Load MCP servers              │
   │ - Setup tools                   │
   └────────────┬──────────────────────┘

2. AGENT LOOP
                │
                ▼
   ┌─────────────────────────────────┐
   │ Process User Input              │
   └────────────┬──────────────────────┘
                │
                ▼
   ┌─────────────────────────────────┐
   │ Send to Claude Model            │
   │ (with full context)             │
   └────────────┬──────────────────────┘
                │
                ▼
   ┌─────────────────────────────────────────────┐
   │ Receive Response from Claude               │
   │ ┌─────────────────┬────────────────────┐   │
   │ │ Type:           │ Action:            │   │
   │ ├─────────────────┼────────────────────┤   │
   │ │ Assistant       │ Stream text output │   │
   │ │ Tool Call       │ Execute tool       │   │
   │ │ Tool Result     │ Return to Claude   │   │
   │ │ Error           │ Handle error       │   │
   │ │ System          │ Manage state       │   │
   │ └─────────────────┴────────────────────┘   │
   └────────────┬──────────────────────────────┘
                │
                ▼
   ┌─────────────────────────────────┐
   │ Decision Point                  │
   │ More to do?                     │
   └────────────┬──────────────────────┘
                │
        ┌───────┴────────┐
        │                │
       YES              NO
        │                │
        ▼                ▼
   Continue Loop    Emit Completion
        │
        └────────────────┬───────────────┐
                         │               │
                    [Go to Process]   [Return Result]

3. STREAM PROCESSING
   For each message from query():
   ┌─────────────────────────────────┐
   │ for await (const message of    │
   │   query(...))                   │
   └────────────┬──────────────────────┘
                │
                ▼
   ┌─────────────────────────────────┐
   │ Check message.type              │
   └────────────┬──────────────────────┘
                │
        ┌───────┼───────┬─────────┬──────┐
        │       │       │         │      │
        ▼       ▼       ▼         ▼      ▼
    assistant tool_call tool_result error system
        │       │       │         │      │
        ▼       ▼       ▼         ▼      ▼
    Process  Execute Collect Handle Update
    Text    Tool     Result  Error  State
```

## Multi-Agent Orchestration Pattern

```
┌──────────────────────────────────────────────────────┐
│        Multi-Agent Orchestration Architecture        │
└──────────────────────────────────────────────────────┘

                    Orchestrator
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
    ┌────────┐    ┌────────┐    ┌────────┐
    │Agent 1 │    │Agent 2 │    │Agent 3 │
    │        │    │        │    │        │
    │Research│    │Analyse │    │Advise  │
    └────┬───┘    └────┬───┘    └────┬───┘
         │             │             │
         ▼             ▼             ▼
    ┌──────────────────────────────────────┐
    │       Shared Context Manager         │
    │                                      │
    │  • Conversation History              │
    │  • Extracted Information             │
    │  • Decision Points                   │
    │  • Resource Allocations              │
    └──────────────────────────────────────┘

Sequential vs Parallel:

SEQUENTIAL:
Agent1 ──► Agent2 ──► Agent3 ──► Final Result
(Context flows left to right)

PARALLEL:
     ┌─► Agent1 ┐
Agent0 ├─► Agent2 ├─► Aggregator ──► Final Result
     └─► Agent3 ┘
(Independent execution, then merge)
```

## Session Management and Forking

```
┌──────────────────────────────────────────────┐
│      Session Management Architecture         │
└──────────────────────────────────────────────┘

                   Initial Query
                        │
                        ▼
            ┌─────────────────────────┐
            │ Session Created         │
            │ ID: abc123def456        │
            │ State: Context + History│
            └────────┬────────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
         ▼           ▼           ▼
    Resume      Continue      Fork
    (existing)  (same)        (branch)
         │           │           │
         ▼           ▼           ▼
    ┌────────┐  ┌────────┐  ┌────────┐
    │Session │  │Session │  │Session │
    │abc123  │  │abc123  │  │def789  │
    │Restored│  │Updated │  │Branched│
    └────────┘  └────────┘  └────────┘

State Compaction:
    Long Conversation → Automatic Compression → 200K Token Context
    ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
    │Recent history│     │Summarise    │     │Maintained   │
    │Old messages  │────▶│Keep summary │────▶│within limit  │
    │Tool outputs  │     │Remove detail│     │Token usage  │
    └──────────────┘     └─────────────┘     └──────────────┘
```

## Tool Execution and MCP Integration

```
┌──────────────────────────────────────────────┐
│    Tool Execution and MCP Architecture       │
└──────────────────────────────────────────────┘

Built-in Tools:
┌─────────────────────────────────────────┐
│ • Read (file)                           │
│ • Write (file)                          │
│ • Edit (file section)                   │
│ • Delete (file/directory)               │
│ • Bash (shell commands)                 │
│ • Grep (text search)                    │
│ • Glob (file patterns)                  │
│ • WebSearch (internet search)           │
│ • Computer Use (mouse, keyboard, vision)│
└─────────────────────────────────────────┘

MCP Servers (Custom Tools):
┌──────────────────────────────────┐
│ createSdkMcpServer()             │
│ ┌───────────────────────────┐    │
│ │ tool 1: Custom Logic      │    │
│ │ tool 2: Database Query    │    │
│ │ tool 3: API Integration   │    │
│ │ tool 4: Business Logic    │    │
│ └───────────────────────────┘    │
└──────────────────────────────────┘

Tool Call Flow:
    Claude ──► Tool Name + Input
                    │
                    ▼
            ┌──────────────────┐
            │ Tool Registry    │
            │ Lookup tool      │
            │ Validate schema  │
            │ Check permission │
            └────────┬─────────┘
                     │
              ┌──────┴──────┐
              │             │
         Allowed        Denied
              │             │
              ▼             ▼
        Execute Tool   Return Error
              │             │
              └──────┬──────┘
                     ▼
            ┌──────────────────┐
            │ Return Result    │
            │ Back to Claude   │
            └──────────────────┘
```

## Permission System Architecture

```
┌──────────────────────────────────────────────┐
│         Permission System Architecture       │
└──────────────────────────────────────────────┘

Permission Modes:

┌─────────────────────────────────────────┐
│ bypassPermissions                       │
│ ✓ No checks, maximum autonomy          │
│ ⚠ Use only for trusted environments    │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ acceptEdits                             │
│ ✓ Auto-approve file modifications      │
│ ⚠ Suitable for development with backup │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ default (Recommended for production)    │
│ ✓ Ask for confirmation on write ops    │
│ ✓ Allow read ops                        │
│ ✓ Customizable via canUseTool callback  │
└─────────────────────────────────────────┘

Permission Decision Flow:
    Tool Requested
            │
            ▼
    ┌──────────────────┐
    │ Check Permission │
    │ Policy           │
    └────────┬─────────┘
             │
        ┌────┴────┬────────┐
        │          │        │
        ▼          ▼        ▼
    Allow      Ask      Deny
     (auto)    (prompt)  (error)
        │          │        │
        └────┬─────┴────┬───┘
             │          │
             ▼          ▼
        Execute    Return Error
        Tool       Or Response
```

## Context Management and Compaction

```
┌──────────────────────────────────────────────┐
│    Context Management and Compaction         │
└──────────────────────────────────────────────┘

Token Budget: 200,000 tokens

As conversation grows:
┌─────────────────────────────────────┐
│ Message 1: User prompt              │  ~50 tokens
├─────────────────────────────────────┤
│ Message 2: Assistant response       │  ~200 tokens
├─────────────────────────────────────┤
│ Message 3: Tool result              │  ~300 tokens
├─────────────────────────────────────┤
│ ...many messages...                 │  ~150,000 tokens
├─────────────────────────────────────┤
│ Token Usage: ~165,000               │
└─────────────────────────────────────┘

Automatic Compaction Triggered:
    When tokens reach threshold (~80% of limit)
            │
            ▼
    ┌─────────────────────────────┐
    │ Analyse conversation        │
    │ - Identify key points       │
    │ - Summarise early messages  │
    │ - Keep recent context       │
    │ - Preserve tool results     │
    └────────────┬────────────────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │ Original: 165,000 tokens    │
    │ Compacted: 85,000 tokens    │
    │ Freed: 80,000 tokens        │
    │ New buffer for continuation │
    └─────────────────────────────┘
```

## Error Handling and Recovery

```
┌────────────────────────────────────┐
│    Error Handling and Recovery     │
└────────────────────────────────────┘

Error Classification:
┌──────────────────────────────────────┐
│ Error Type          │ Severity       │
├──────────────────────┼────────────────┤
│ Authentication      │ CRITICAL       │
│ Permission Denied   │ HIGH           │
│ Budget Exceeded     │ HIGH           │
│ Timeout             │ MEDIUM         │
│ Rate Limit          │ MEDIUM         │
│ Tool Failure        │ LOW-MEDIUM     │
│ Invalid Input       │ LOW            │
└──────────────────────────────────────┘

Error Recovery Flow:
    Error Occurs
            │
            ▼
    ┌──────────────────────┐
    │ Classify Error       │
    │ Determine Severity   │
    └────────────┬─────────┘
                 │
        ┌────────┼────────┐
        │        │        │
        ▼        ▼        ▼
    Critical  High    Medium/Low
        │        │        │
        ▼        ▼        ▼
    Stop    Check   Retry
    Log     Log     Fallback
    Alert   Alert   Log
        │        │        │
        └────────┼────────┘
                 │
                 ▼
         ┌──────────────┐
         │ Return Error │
         │ to Client    │
         └──────────────┘

Retry with Exponential Backoff:
Attempt 1: 1 second
    ↓ (if fails)
Attempt 2: 2 seconds
    ↓ (if fails)
Attempt 3: 4 seconds
    ↓ (if fails)
Attempt 4: 8 seconds
    ↓ (if all fail)
Return Failure
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────┐
│         Complete Data Flow Diagram          │
└─────────────────────────────────────────────┘

User Input
    │
    ▼
┌──────────────────┐
│ Input Validation │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Create Query     │
│ - Prompt         │
│ - Options        │
│ - Context        │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────┐
│ Agent Execution Loop             │
│                                  │
│ For each message:                │
│                                  │
│ ┌─────────────────────────────┐  │
│ │ if (message.type ==)        │  │
│ │   'assistant': Stream text  │  │
│ │   'tool_call': Execute tool │  │
│ │   'tool_result': To context │  │
│ │   'error': Handle error     │  │
│ │   'system': Update state    │  │
│ └──────────────┬──────────────┘  │
│                │                  │
│                ▼                  │
│        Check loop condition      │
│        Continue or exit?          │
└────────────┬─────────────────────┘
             │
         ┌───┴────┐
         │        │
      More    Done
        │        │
        │        ▼
        │   ┌──────────────┐
        │   │ Aggregation  │
        │   │ Final result │
        │   └──────────────┘
        │        │
        └────────┘
             │
             ▼
      ┌─────────────────┐
      │ Output response │
      │ to user         │
      └─────────────────┘
```

## Performance and Scaling Architecture

```
┌──────────────────────────────────┐
│  Performance & Scaling Diagram   │
└──────────────────────────────────┘

Single Instance:
    User ──► Agent Instance ──► Claude API

Scaled Deployment:
    ┌─────────────────────────────────────┐
    │         Load Balancer               │
    │       (Round Robin / Sticky)        │
    └────────────┬────────────────────────┘
                 │
    ┌────────────┼────────────────┐
    │            │                │
    ▼            ▼                ▼
  Instance    Instance         Instance
  (with       (with            (with
   Cache)      Cache)           Cache)
    │            │                │
    └────────────┼────────────────┘
                 │
    ┌────────────▼────────────────┐
    │      Claude API (Scaled)     │
    │  - Rate limiting             │
    │  - Load distribution         │
    │  - Failover handling         │
    └─────────────────────────────┘

Caching Strategy:
    Prompt Input
         │
         ▼
    ┌─────────────────┐
    │ Check L1 Cache  │ (In-process)
    └────────┬────────┘
             │
        Not Found
             │
             ▼
    ┌─────────────────┐
    │ Check L2 Cache  │ (Redis)
    └────────┬────────┘
             │
        Not Found
             │
             ▼
    ┌─────────────────┐
    │ Query Claude    │
    │ Cache Result    │
    └─────────────────┘
```

These diagrams illustrate the complex interactions and architectural patterns within the Claude Agent SDK. Each represents a different aspect of the system's operation, from basic execution flows to advanced orchestration patterns.
